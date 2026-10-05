import { useState } from "react";
import { lessons } from "./lessons/index.js";

function Scenario({ scenario }) {
  const [revealed, setRevealed] = useState(false);

  return (
    <div className="scenario">
      <div className="scenario-header">
        <span>{scenario.label}</span>
        <button type="button" onClick={() => setRevealed((r) => !r)}>
          {revealed ? "Hide output" : "Reveal output"}
        </button>
      </div>
      {revealed && <div className="scenario-output">{scenario.render()}</div>}
    </div>
  );
}

export default function App() {
  const [lessonId, setLessonId] = useState(lessons.at(-1).id);
  const lesson = lessons.find((l) => l.id === lessonId);

  return (
    <main className="app">
      <header>
        <h1>React Practice</h1>
        <select value={lessonId} onChange={(e) => setLessonId(e.target.value)}>
          {lessons.map((l) => (
            <option key={l.id} value={l.id}>
              {l.title}
            </option>
          ))}
        </select>
      </header>

      <p className="hint">
        Read the code and <code>QUESTIONS.md</code> in{" "}
        <code>src/lessons/{lesson.id}/</code>. Predict the output first, then reveal it.
      </p>

      {lesson.scenarios.map((scenario) => (
        <Scenario key={`${lesson.id}-${scenario.label}`} scenario={scenario} />
      ))}
    </main>
  );
}
