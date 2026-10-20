import { useState } from "react";

function Cart() {
  const [cart, setCart] = useState([
    { id: "a", name: "Shoe", qty: 1 },
    { id: "b", name: "Hat", qty: 2 },
  ]);

  function removeItem(id) {
    setCart(cart.filter((item) => item.id !== id));
  }

  function increaseQty(id) {
    setCart(
      cart.map((item) =>
        item.id === id ? { ...item, qty: item.qty + 1 } : item
      )
    );
  }

  function increaseQtyInPlace(id) {
    const item = cart.find((item) => item.id === id);
    item.qty = item.qty + 1;
    setCart(cart);
  }

  return (
    <ul className="cart">
      {cart.map((item) => (
        <li key={item.id}>
          <span>
            {item.name} × {item.qty}
          </span>
          <button type="button" onClick={() => increaseQty(item.id)}>
            +1
          </button>
          <button type="button" onClick={() => increaseQtyInPlace(item.id)}>
            +1 in place
          </button>
          <button type="button" onClick={() => removeItem(item.id)}>
            Remove
          </button>
        </li>
      ))}
    </ul>
  );
}

// Used like this:
const app = <Cart />;
