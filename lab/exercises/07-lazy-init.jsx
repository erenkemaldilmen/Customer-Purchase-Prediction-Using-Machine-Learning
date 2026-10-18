import { useState } from "react";

function loadCart(label) {
  console.log(label + ": reading the cart from localStorage");
  const saved = localStorage.getItem("cart");
  return saved ? JSON.parse(saved) : [];
}

function EagerCart() {
  const [items] = useState(loadCart("Eager"));
  const [clicks, setClicks] = useState(0);

  return (
    <div className="cart-demo">
      <strong>Eager</strong>
      <span>{items.length} items</span>
      <button type="button" onClick={() => setClicks(clicks + 1)}>
        Re-render ({clicks})
      </button>
    </div>
  );
}

function LazyCart() {
  const [items] = useState(() => loadCart("Lazy"));
  const [clicks, setClicks] = useState(0);

  return (
    <div className="cart-demo">
      <strong>Lazy</strong>
      <span>{items.length} items</span>
      <button type="button" onClick={() => setClicks(clicks + 1)}>
        Re-render ({clicks})
      </button>
    </div>
  );
}

function CartDemo() {
  return (
    <div>
      <EagerCart />
      <LazyCart />
    </div>
  );
}

// Used like this:
const app = <CartDemo />;
