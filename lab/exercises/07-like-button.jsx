import { useState } from "react";

function LikeButton({ initialLikes }) {
  const [likes, setLikes] = useState(initialLikes);
  const [liked, setLiked] = useState(false);

  function handleClick() {
    if (liked) {
      setLikes(likes - 1);
    } else {
      setLikes(likes + 1);
    }
    setLiked(!liked);
  }

  return (
    <button
      type="button"
      className={liked ? "liked" : ""}
      onClick={handleClick}
    >
      {liked ? "♥" : "♡"} {likes}
    </button>
  );
}

function CartBadge() {
  const [items] = useState(() => {
    const saved = localStorage.getItem("cart");
    return saved ? JSON.parse(saved) : [];
  });

  return <span className="badge">{items.length} items in cart</span>;
}

function PostFooter() {
  return (
    <div>
      <LikeButton initialLikes={10} />
      <LikeButton initialLikes={3} />
      <CartBadge />
    </div>
  );
}

// Used like this:
const app = <PostFooter />;
