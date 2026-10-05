import { Inbox } from "./Inbox.jsx";

const kemal = { name: "Kemal" };
const threeMessages = [
  { id: 1, subject: "Invoice", read: true },
  { id: 2, subject: "Meeting", read: false },
  { id: 3, subject: "Hello", read: true },
];

export const lesson = {
  id: "04-conditional-rendering",
  title: "4. Conditional rendering",
  scenarios: [
    {
      label: "Q1: loading AND error at the same time",
      render: () => (
        <Inbox
          user={kemal}
          messages={threeMessages}
          isLoading={true}
          error={{ message: "Network down" }}
        />
      ),
    },
    {
      label: "Q2: Scenario A (3 messages, 1 unread)",
      render: () => (
        <Inbox user={kemal} messages={threeMessages} isLoading={false} error={null} />
      ),
    },
    {
      label: "Q3: Scenario B (no messages)",
      render: () => <Inbox user={kemal} messages={[]} isLoading={false} error={null} />,
    },
    {
      label: "Q4: what `{0 && ...}` puts on screen",
      render: () => (
        <div>
          <p>With <code>{"{unreadCount > 0 && ...}"}</code>: [{0 > 0 && <b>notice</b>}]</p>
          <p>With <code>{"{unreadCount && ...}"}</code>: [{0 && <b>notice</b>}]</p>
        </div>
      ),
    },
  ],
};
