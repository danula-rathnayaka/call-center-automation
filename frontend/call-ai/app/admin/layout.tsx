import { NavBar } from "../components/admin/Navbar";

export default function AdminLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <>
      <NavBar />
      <div className="pt-12 px-80 bg-neutral-50">{children}</div>
    </>
  );
}
