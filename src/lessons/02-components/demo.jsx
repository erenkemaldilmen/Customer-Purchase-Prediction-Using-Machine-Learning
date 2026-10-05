import { TeamList } from "./TeamList.jsx";

export const lesson = {
  id: "02-components",
  title: "2. Components",
  scenarios: [
    {
      label: "Ayşe (admin) and Mehmet (editor)",
      render: () => (
        <TeamList
          members={[
            { id: 1, name: "Ayşe", role: "admin", photoUrl: "/a.png" },
            { id: 2, name: "Mehmet", role: "editor", photoUrl: "/m.png" },
          ]}
        />
      ),
    },
  ],
};
