# React Practice Track

Format: Claude shows clean, production-quality code (the kind AI tools produce).
The goal is to understand it: what it renders, how data flows, what happens
step by step, and why it is written that way. No bug hunting, no writing code.

Progress: `[ ]` not started · `[~]` in progress · `[x]` done

## Level 1 — Foundations
- [x] 1. JSX basics: expressions, attributes, `className`, self-closing tags, fragments
- [x] 2. Components: function components, naming, composition, returning `null`
- [x] 3. Props: passing data, destructuring, defaults, `children`, read-only props
- [x] 4. Conditional rendering: `&&`, ternary, early return, the `0 &&` trap
- [~] 5. Lists & keys: `map`, why keys matter, index-as-key problems
- [ ] 6. Events: handlers, passing vs calling functions, `e.preventDefault`, synthetic events

## Level 2 — State & Rendering
- [ ] 7. `useState`: initial value, setter, lazy initializer
- [ ] 8. State updates are async/batched: stale values, functional updates `setX(prev => ...)`
- [ ] 9. Immutability: updating objects and arrays in state
- [ ] 10. Render cycle: when components re-render, render vs commit, pure rendering
- [ ] 11. Controlled vs uncontrolled inputs, forms
- [ ] 12. Lifting state up & single source of truth
- [ ] 13. Derived state: computing during render instead of syncing state

## Level 3 — Effects & Refs
- [ ] 14. `useEffect`: dependency array, run timing, cleanup
- [ ] 15. Common effect bugs: missing deps, infinite loops, stale closures
- [ ] 16. Data fetching in effects: race conditions, AbortController, loading/error states
- [ ] 17. "You might not need an effect"
- [ ] 18. `useRef`: DOM refs, mutable values that don't trigger renders
- [ ] 19. `useLayoutEffect` vs `useEffect`
- [ ] 20. StrictMode double-invoking and why

## Level 4 — Advanced Hooks & Patterns
- [ ] 21. `useReducer`: actions, reducers, when to prefer it over `useState`
- [ ] 22. Context: `createContext`, Provider, `useContext`, re-render cost
- [ ] 23. Custom hooks: extracting logic, rules of hooks
- [ ] 24. `useMemo` & `useCallback`: referential equality, when they help / don't
- [ ] 25. `React.memo` and avoiding unnecessary re-renders
- [ ] 26. `useId`, `useImperativeHandle`, `forwardRef` / ref as a prop
- [ ] 27. Component patterns: composition, render props, HOCs, compound components
- [ ] 28. State identity: same position = same state, resetting state with `key`

## Level 5 — Modern React (18 / 19)
- [ ] 29. Concurrent rendering: `useTransition`, `useDeferredValue`
- [ ] 30. Suspense & `lazy` code-splitting
- [ ] 31. Error boundaries
- [ ] 32. Portals
- [ ] 33. React 19: `use`, Actions, `useActionState`, `useFormStatus`, `useOptimistic`
- [ ] 34. Server Components vs Client Components, `"use client"` / `"use server"`
- [ ] 35. React Compiler and what it changes about memoization

## Level 6 — Ecosystem & Practice
- [ ] 36. Routing (React Router): routes, params, nested layouts, navigation
- [ ] 37. Server state (TanStack Query): caching, invalidation
- [ ] 38. Global state options: Context vs Zustand vs Redux Toolkit
- [ ] 39. Styling approaches: CSS modules, Tailwind, CSS-in-JS
- [ ] 40. TypeScript with React: typing props, events, hooks, generics
- [ ] 41. Testing: React Testing Library, user-centric queries, async tests
- [ ] 42. Performance: profiling, virtualization, bundle size
- [ ] 43. Accessibility: semantic HTML, labels, focus management
- [ ] 44. Project structure & code review: spotting bugs in real-world components
