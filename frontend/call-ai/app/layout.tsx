import type { Metadata } from "next";
import { Poppins } from "next/font/google";
import "./globals.css";
import { NotificationProvider } from "./components/notifications/NotificationProvider";

const font = Poppins({
  variable: "--font-poppins",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Call Center AI",
  description: "Call Center AI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${font.variable} antialiased`}>
        <NotificationProvider>{children}</NotificationProvider>
      </body>
    </html>
  );
}
