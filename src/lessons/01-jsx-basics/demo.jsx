import { ProductCard } from "./ProductCard.jsx";

export const lesson = {
  id: "01-jsx-basics",
  title: "1. JSX basics",
  scenarios: [
    {
      label: "Desk Lamp (out of stock)",
      render: () => (
        <ProductCard
          product={{ name: "Desk Lamp", price: 24.5, tags: ["home", "light"], inStock: false }}
        />
      ),
    },
    {
      label: "Notebook (in stock, no tags)",
      render: () => (
        <ProductCard product={{ name: "Notebook", price: 3, tags: [], inStock: true }} />
      ),
    },
  ],
};
