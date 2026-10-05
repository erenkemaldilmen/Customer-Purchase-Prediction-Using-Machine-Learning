# 2. Components — `TeamList.jsx`

Status: done

## Questions
1. Draw the component tree, starting from `TeamList`.
2. What shows on screen for Ayşe and for Mehmet? What's different, and why?
3. What does it mean for a component to return `null`?
4. `UserCard` never passes `size`. How big is the avatar, and why?
5. Why split this into 4 small components instead of one big one?

## Review notes
```
TeamList
├── UserCard (Ayşe)
│   ├── Avatar
│   └── Badge  → "Admin"
└── UserCard (Mehmet)
    ├── Avatar
    └── Badge  → null
```
- One component (`UserCard`) appears many times with different data: one template, many instances.
- `Badge` decides by itself whether to show anything, so the parent stays simple.
- A default value (`size = 48`) is used only when the prop is `undefined`, not for `null` or `0`.
- Splitting is separation of concerns: reuse, readability, and changes stay in one place.

(The photos are placeholder paths, so the browser shows the `alt` text instead of an image.)
