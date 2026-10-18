# React Event Lab

The interactive page that runs each practice exercise and traces what happens:
clicks and bubbling, handlers, `stopPropagation` / `preventDefault`, function
props called by children, `useState` set calls, re-renders, and `console.log`.

Published at: https://claude.ai/artifact/7tL6uABSdWPwXt6U38FCAw

## Build

```bash
npm install
npm run lab        # writes lab/dist/react-event-lab.html
```

## Layout

```
lab/
  build.cjs        compiles exercises with esbuild, highlights them with Prism,
                   and fills template.html
  template.html    the page: styles, tracing runtime, UI
  exercises/       one .jsx file per exercise (no `export`; end with `const app = <... />`)
```

## Adding an exercise

1. Add `exercises/NN-name.jsx`. Write components as top-level `function Name(...)`
   declarations and finish with `const app = <Root />;`.
2. Add an entry at the top of the `exercises` list in `build.cjs`
   (the first entry is the default tab).
3. `npm run lab`, then republish `lab/dist/react-event-lab.html` to the same URL.

## Limits

- Runs React 18.3.1 from cdnjs (React 19 has no UMD build). React 19-only APIs need a bundled React.
- Browser only: no Next.js server features.
- Traces `useState`; other hooks run normally but are not traced yet.
