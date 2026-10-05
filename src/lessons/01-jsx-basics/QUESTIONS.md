# 1. JSX basics — `ProductCard.jsx`

Status: done

## Questions
1. In plain words, what does the component show for the `Desk Lamp` product? Top to bottom.
2. What are `<>` and `</>`, and why does this component need them?
3. What do the `{ }` curly braces mean in JSX?
4. What does `tags.length > 0 && (...)` say? What shows if `tags` is empty?
5. Why does each `<li>` get a `key`?

## Review notes
- The price shows as **$24.50**: `toFixed(2)` gives `"24.50"` and the template string adds `$`.
- A fragment groups several siblings into the one root a component must return,
  without adding an extra element to the page. It has nothing to do with props.
- `{ }` switches to JavaScript, but only **expressions** (values) fit inside.
  That's why JSX uses a ternary instead of `if` and `.map()` instead of `for`.
- An empty `tags` array makes the condition `false`, so the whole `<ul>` disappears.
- `key` is required by **React** (not JavaScript): it's how React tells list items
  apart between renders when items are added, removed, or reordered.
