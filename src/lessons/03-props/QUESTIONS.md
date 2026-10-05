# 3. Props — `OrderSummary.jsx`

Status: done

## Questions
1. `OrderSummary` never writes `children={...}`. What is `children` inside `Card`, and where does it come from?
2. `footer` holds a whole `<Button />`. What does that tell you about what a prop can hold?
3. Inside `Button`, what is in `rest`? What does `{...rest}` do?
4. What is the outer `div`'s `className`? What's the total, and is Pay clickable?
5. Why should `Card` never do `title = title.toUpperCase()`? Who owns `title`?

## Review notes
- `children` is whatever sits between `<Card>` and `</Card>`. React fills it in automatically.
- A prop can hold **any** JavaScript value: string, number, object, function, or JSX.
- `rest = { onClick, disabled }`. `label` was taken out by name. `{...rest}` forwards the leftovers to `<button>`.
- className is `card card--highlight` (two dashes). Total is 25 (10×2 + 5×1). The button is enabled.
- In `<p>Total: ${total}</p>` the `$` is a plain dollar sign on screen; only `{total}` is JSX.
  Inside backticks, `${...}` is template string syntax. Same look, different meaning.
- Props are owned by the parent and flow one way, down. Treat them as read-only;
  if you need a different version, make a new variable (`const upperTitle = title.toUpperCase()`).
