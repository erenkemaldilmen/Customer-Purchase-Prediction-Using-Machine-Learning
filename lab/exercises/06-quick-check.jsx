function Card() {
  return (
    <div onClick={() => console.log("card")}>
      <h3>Title</h3>
      <button onClick={() => console.log("like")}>Like</button>
      <button
        onClick={(e) => {
          e.stopPropagation();
          console.log("share");
        }}
      >
        Share
      </button>
    </div>
  );
}

// Used like this:
const app = <Card />;
