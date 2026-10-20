import { useState } from "react";

function Header({ title }) {
  return <h2>{title}</h2>;
}

function ThemeLabel({ dark }) {
  return <p>Theme: {dark ? "dark" : "light"}</p>;
}

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <button type="button" onClick={() => setCount(count + 1)}>
      Clicked {count} times
    </button>
  );
}

function Dashboard() {
  const [dark, setDark] = useState(false);

  return (
    <div className="dashboard">
      <Header title="My dashboard" />
      <ThemeLabel dark={dark} />
      <button type="button" onClick={() => setDark(!dark)}>
        Toggle theme
      </button>
      <Counter />
    </div>
  );
}

// Used like this:
const app = <Dashboard />;
