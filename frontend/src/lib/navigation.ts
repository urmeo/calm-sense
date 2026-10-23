export function containNavigationFocus(sidebar: HTMLElement, background: HTMLElement, onClose: () => void) {
  const document = sidebar.ownerDocument;
  const previousFocus = document.activeElement;
  const previousInert = background.inert;
  const controls = () => Array.from(sidebar.querySelectorAll<HTMLElement>('a[href], button:not(:disabled)'));
  background.inert = true;
  controls()[0]?.focus();

  const keepFocus = () => {
    if (!sidebar.contains(document.activeElement)) controls()[0]?.focus();
  };
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      onClose();
    } else if (event.key === 'Tab') {
      const items = controls();
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    }
  };
  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('focusin', keepFocus);
  return () => {
    document.removeEventListener('keydown', onKeyDown);
    document.removeEventListener('focusin', keepFocus);
    background.inert = previousInert;
    if (previousFocus instanceof HTMLElement && previousFocus.isConnected) previousFocus.focus();
  };
}
