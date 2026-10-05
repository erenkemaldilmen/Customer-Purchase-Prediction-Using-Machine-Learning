export function ProductCard({ product }) {
  const { name, price, tags, inStock } = product;
  const formattedPrice = `$${price.toFixed(2)}`;

  return (
    <>
      <h2 className="product-title">{name}</h2>
      <p>{formattedPrice}</p>

      {inStock ? (
        <button type="button">Add to cart</button>
      ) : (
        <span className="sold-out">Sold out</span>
      )}

      {tags.length > 0 && (
        <ul>
          {tags.map((tag) => (
            <li key={tag}>{tag}</li>
          ))}
        </ul>
      )}
    </>
  );
}
