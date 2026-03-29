import { ClerkProvider } from "@clerk/nextjs";
import { NotificationProvider } from "../components/notifications/NotificationProvider";

export default async function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ClerkProvider>
      <NotificationProvider>
        <div className="px-80 bg-neutral-50">{children}</div>
      </NotificationProvider>
    </ClerkProvider>
  );
}
