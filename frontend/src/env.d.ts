interface ImportMeta {
  readonly env: { readonly BASE_URL: string };
}

declare module '*.css';

declare module '*.svg' {
  const source: string;
  export default source;
}
