function Avatar({ src, name, size = 48 }) {
  return (
    <img
      className="avatar"
      src={src}
      alt={name}
      width={size}
      height={size}
    />
  );
}

function Badge({ role }) {
  if (role !== "admin") return null;
  return <span className="badge">Admin</span>;
}

function UserCard({ user }) {
  return (
    <div className="user-card">
      <Avatar src={user.photoUrl} name={user.name} />
      <div>
        <strong>{user.name}</strong>
        <Badge role={user.role} />
      </div>
    </div>
  );
}

export function TeamList({ members }) {
  return (
    <section>
      {members.map((member) => (
        <UserCard key={member.id} user={member} />
      ))}
    </section>
  );
}
