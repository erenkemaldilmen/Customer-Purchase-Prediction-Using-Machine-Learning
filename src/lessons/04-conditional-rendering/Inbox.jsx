function Spinner() {
  return <p className="spinner">Loading…</p>;
}

function ErrorMessage({ text }) {
  return <p className="error">Something went wrong: {text}</p>;
}

export function Inbox({ user, messages, isLoading, error }) {
  if (isLoading) return <Spinner />;
  if (error) return <ErrorMessage text={error.message} />;
  if (!user) return <p>Please log in to see your inbox.</p>;

  const unreadCount = messages.filter((m) => !m.read).length;

  return (
    <div>
      <h2>Welcome back, {user.name}</h2>

      {unreadCount > 0 && (
        <p className="notice">You have {unreadCount} unread messages</p>
      )}

      {messages.length === 0 ? (
        <p>Your inbox is empty.</p>
      ) : (
        <ul>
          {messages.map((m) => (
            <li key={m.id} className={m.read ? "read" : "unread"}>
              {m.subject}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
