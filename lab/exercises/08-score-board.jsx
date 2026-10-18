import { useState } from "react";

function ScoreBoard() {
  const [score, setScore] = useState(0);

  function addThreeDirect() {
    setScore(score + 1);
    setScore(score + 1);
    setScore(score + 1);
  }

  function addThreeUpdater() {
    setScore((s) => s + 1);
    setScore((s) => s + 1);
    setScore((s) => s + 1);
  }

  function addTenAndLog() {
    setScore(score + 10);
    console.log("score right after setScore:", score);
  }

  return (
    <div className="scoreboard">
      <p className="score">Score: {score}</p>
      <button type="button" onClick={addThreeDirect}>
        +3 (direct)
      </button>
      <button type="button" onClick={addThreeUpdater}>
        +3 (updater)
      </button>
      <button type="button" onClick={addTenAndLog}>
        +10 and log
      </button>
    </div>
  );
}

// Used like this:
const app = <ScoreBoard />;
