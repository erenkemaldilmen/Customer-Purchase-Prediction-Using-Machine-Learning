import { useState } from "react";

function TodoList() {
  const [todos, setTodos] = useState([
    { id: 1, text: "Buy milk", done: false },
    { id: 2, text: "Learn React", done: true },
  ]);
  const [nextId, setNextId] = useState(3);

  function addTodo() {
    setTodos([...todos, { id: nextId, text: "New task " + nextId, done: false }]);
    setNextId(nextId + 1);
  }

  function toggleTodo(id) {
    setTodos(
      todos.map((todo) =>
        todo.id === id ? { ...todo, done: !todo.done } : todo
      )
    );
  }

  function removeTodo(id) {
    setTodos(todos.filter((todo) => todo.id !== id));
  }

  function addTodoByPush() {
    todos.push({ id: Date.now(), text: "Pushed item", done: false });
    setTodos(todos);
  }

  return (
    <div className="todos">
      <ul>
        {todos.map((todo) => (
          <li key={todo.id} className={todo.done ? "done" : ""}>
            <label>
              <input
                type="checkbox"
                checked={todo.done}
                onChange={() => toggleTodo(todo.id)}
              />
              {todo.text}
            </label>
            <button type="button" onClick={() => removeTodo(todo.id)}>
              Remove
            </button>
          </li>
        ))}
      </ul>
      <button type="button" onClick={addTodo}>
        Add task
      </button>
      <button type="button" onClick={addTodoByPush}>
        Add with push
      </button>
    </div>
  );
}

// Used like this:
const app = <TodoList />;
