function Card({ title, footer, children, variant = "default" }) {
  return (
    <div className={`card card--${variant}`}>
      <h3>{title}</h3>
      <div className="card-body">{children}</div>
      {footer && <div className="card-footer">{footer}</div>}
    </div>
  );
}

function Button({ label, ...rest }) {
  return (
    <button type="button" {...rest}>
      {label}
    </button>
  );
}

export function OrderSummary({ order }) {
  const total = order.items.reduce((sum, item) => sum + item.price * item.qty, 0);

  return (
    <Card
      title={`Order #${order.id}`}
      variant="highlight"
      footer={
        <Button
          label="Pay now"
          onClick={() => alert("Paying...")}
          disabled={total === 0}
        />
      }
    >
      <p>{order.items.length} items</p>
      <p>Total: ${total}</p>
    </Card>
  );
}
