from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.multiclass import check_classification_targets


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, dropout: float = 0.2):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(2)
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return self.pool(self.drop(out))


class _Net(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, widths=(32, 64, 128)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, widths[0], kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        blocks = []
        prev = widths[0]
        for w in widths:
            blocks.append(_ResidualBlock(prev, w))
            prev = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(prev, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


class CNN1DClassifier:
    def __init__(
        self,
        in_channels: int = 5,
        max_epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        patience: int = 8,
        val_fraction: float = 0.15,
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        self.in_channels = in_channels
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Any = None
        self.classes_: Any = None
        self._mean: Any = None
        self._std: Any = None

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if (
            not np.isfinite(self._mean).all()
            or not np.isfinite(self._std).all()
            or np.any(self._std <= 0)
        ):
            raise ValueError(
                "CNN normalization requires finite means and positive finite scales"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            standardized = (x - self._mean) / self._std
        if not np.isfinite(standardized).all():
            raise ValueError("CNN normalization produced nonfinite windows")
        return standardized

    def _validate_windows(
        self, X: np.ndarray, *, allow_empty: bool = False
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3 or X.shape[1] != self.in_channels or X.shape[2] < 61:
            raise ValueError(
                "CNN windows require shape (samples, in_channels, at least 61 timesteps)"
            )
        if (not len(X) and not allow_empty) or not np.isfinite(X).all():
            raise ValueError("CNN windows must be nonempty and finite")
        return X

    def _validation_split(self, y: np.ndarray, groups=None):
        indices = np.arange(len(y))
        if groups is None:
            return train_test_split(
                indices,
                test_size=self.val_fraction,
                stratify=y,
                random_state=self.random_state,
            )
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(y) or len(np.unique(groups)) < 2:
            raise ValueError(
                "CNN subject validation requires at least two training subjects"
            )
        classes = np.unique(y)
        splitter = GroupShuffleSplit(
            n_splits=20, test_size=self.val_fraction, random_state=self.random_state
        )
        for train, validation in splitter.split(indices, y, groups):
            if np.array_equal(np.unique(y[train]), classes):
                return train, validation

        subjects, group_index = np.unique(groups, return_inverse=True)
        n_train = len(subjects) - int(np.ceil(self.val_fraction * len(subjects)))
        class_index = np.searchsorted(classes, y)
        class_masks = [
            sum(
                1 << int(label)
                for label in np.unique(class_index[group_index == subject])
            )
            for subject in range(len(subjects))
        ]
        order = np.random.RandomState(self.random_state).permutation(len(subjects))
        # Class coverage
        class_coverage: dict[int, tuple[int, ...]] = {0: ()}
        all_classes_mask = (1 << len(classes)) - 1
        for subject in order:
            for covered_classes, selected in list(class_coverage.items()):
                if len(selected) >= n_train:
                    continue
                next_mask = covered_classes | class_masks[subject]
                candidate = (*selected, int(subject))
                if next_mask not in class_coverage or len(candidate) < len(
                    class_coverage[next_mask]
                ):
                    class_coverage[next_mask] = candidate
            if all_classes_mask in class_coverage:
                break
        if all_classes_mask not in class_coverage:
            raise ValueError(
                "No subject validation split can retain every training class at this fraction"
            )
        training_groups = list(class_coverage[all_classes_mask])
        training_groups += [
            int(subject) for subject in order if subject not in training_groups
        ][: n_train - len(training_groups)]
        training = np.isin(group_index, training_groups)
        return indices[training], indices[~training]

    def fit(self, X: np.ndarray, y: np.ndarray, groups=None) -> "CNN1DClassifier":
        """Grouped validation subjects stay disjoint."""
        self.model = None
        self.classes_ = None
        self._mean = self._std = None
        for name in ("in_channels", "max_epochs", "batch_size", "patience"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.val_fraction < 1 or not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError(
                "val_fraction must be in (0, 1) and lr must be finite and positive"
            )
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

        X = self._validate_windows(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("CNN labels must have one entry per window")
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("CNN training requires at least two classes")
        y_idx = np.searchsorted(self.classes_, y)

        train, validation = self._validation_split(y_idx, groups)
        if len(train) < 2:
            raise ValueError(
                "CNN fitting requires at least two windows after validation splitting"
            )
        self._mean = X[train].mean(axis=(0, 2), keepdims=True)
        self._std = X[train].std(axis=(0, 2), keepdims=True) + 1e-8
        x_tr, x_val = self._standardize(X[train]), self._standardize(X[validation])
        y_tr, y_val = y_idx[train], y_idx[validation]

        weights = compute_class_weight(
            "balanced", classes=np.arange(len(self.classes_)), y=y_tr
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device)
        )

        model = _Net(self.in_channels, len(self.classes_)).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.max_epochs
        )

        tr = torch.utils.data.TensorDataset(
            torch.from_numpy(x_tr), torch.from_numpy(y_tr).long()
        )
        gen = torch.Generator().manual_seed(self.random_state)
        loader = torch.utils.data.DataLoader(
            tr, batch_size=self.batch_size, shuffle=True, generator=gen
        )
        x_val_t = torch.from_numpy(x_val).to(self.device)
        y_val_t = torch.from_numpy(y_val).long().to(self.device)

        best_loss = np.inf
        best_state = None
        stale = 0

        for _ in range(self.max_epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                if not torch.isfinite(loss):
                    raise ValueError("CNN training produced a nonfinite loss")
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(x_val_t), y_val_t).item()
            if not np.isfinite(val_loss):
                raise ValueError("CNN validation produced a nonfinite loss")
            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.classes_ is None:
            raise NotFittedError("Fit CNN1DClassifier before predicting")
        X = self._validate_windows(X, allow_empty=True)
        if not len(X):
            return np.empty((0, len(self.classes_)), dtype=np.float32)
        X = self._standardize(X)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(X[i : i + self.batch_size]).to(self.device)
                probs.append(torch.softmax(self.model(xb), dim=1).cpu().numpy())
        proba = np.concatenate(probs, axis=0)
        if not np.isfinite(proba).all():
            raise ValueError("CNN inference produced nonfinite probabilities")
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
