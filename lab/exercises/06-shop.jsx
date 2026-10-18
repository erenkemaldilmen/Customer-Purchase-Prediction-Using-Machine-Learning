function ProductRow({ product, onAddToCart, onRemove }) {
  return (
    <li
      className="product-row"
      onClick={() => console.log("Row clicked:", product.name)}
    >
      <span>{product.name}</span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onAddToCart(product.id);
        }}
      >
        Add
      </button>
      <button type="button" onClick={() => onRemove(product.id)}>
        Remove
      </button>
    </li>
  );
}

function SearchForm({ onSearch }) {
  function handleSubmit(e) {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    onSearch(formData.get("query").trim());
  }

  return (
    <form onSubmit={handleSubmit}>
      <input name="query" placeholder="Search products" />
      <button type="submit">Search</button>
    </form>
  );
}

function Shop({ products }) {
  function handleAddToCart(id) {
    console.log("Added", id);
  }
  function handleRemove(id) {
    console.log("Removed", id);
  }
  function handleSearch(query) {
    console.log("Searching for", query);
  }

  return (
    <>
      <SearchForm onSearch={handleSearch} />
      <ul>
        {products.map((p) => (
          <ProductRow
            key={p.id}
            product={p}
            onAddToCart={handleAddToCart}
            onRemove={handleRemove}
          />
        ))}
      </ul>
    </>
  );
}

// Used like this:
const app = (
  <Shop products={[{ id: "p1", name: "Mouse" }, { id: "p2", name: "Keyboard" }]} />
);
