import { OrderSummary } from "./OrderSummary.jsx";

export const lesson = {
  id: "03-props",
  title: "3. Props",
  scenarios: [
    {
      label: "Order #1042",
      render: () => (
        <OrderSummary
          order={{ id: 1042, items: [{ price: 10, qty: 2 }, { price: 5, qty: 1 }] }}
        />
      ),
    },
    {
      label: "Empty order #1043",
      render: () => <OrderSummary order={{ id: 1043, items: [] }} />,
    },
  ],
};
