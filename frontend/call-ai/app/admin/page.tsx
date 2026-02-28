"use client";

import { useMemo } from "react";

export default function AdminPage() {
  const today = useMemo(() => {
    return new Date().toLocaleDateString("en-US", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  }, []);

  return (
    <div className="flex min-h-screen bg-neutral-50 text-neutral-800">
      {/* Main */}
      <main className="flex-1 p-10 border-x border-neutral-200">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold">Welcome back, Arosha</h1>
            <p className="text-sm text-neutral-500 mt-1">{today}</p>
          </div>
        </div>

        {/* Quick Action*/}
        <div className="mt-14">
          <h2 className="text-lg font-semibold mb-6">Quick Actions</h2>

          <div className="flex gap-6">
            <ActionCard title="Calls" iconType="phone" />
            <ActionCard title="Documents" iconType="doc" />
          </div>
        </div>

        {/* Recent Calls */}
        <div className="mt-16">
          <h2 className="text-lg font-semibold mb-6">Recent Calls</h2>

          <div className="bg-white rounded-xl border border-neutral-200 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-neutral-100 text-neutral-600">
                <tr>
                  <th className="text-left px-6 py-4">Caller</th>
                  <th className="text-left px-6 py-4">Phone</th>
                  <th className="text-left px-6 py-4">Duration</th>
                  <th className="text-left px-6 py-4">Status</th>
                  <th className="text-left px-6 py-4">Date</th>
                </tr>
              </thead>
              <tbody>
                <CallRow
                  name="John Perera"
                  phone="+94 77 123 4567"
                  duration="4m 12s"
                  status="Completed"
                  date="Today"
                />
                <CallRow
                  name="Nimali Silva"
                  phone="+94 71 998 4455"
                  duration="—"
                  status="Missed"
                  date="Today"
                />
                <CallRow
                  name="Kasun Fernando"
                  phone="+94 76 111 2233"
                  duration="2m 03s"
                  status="Completed"
                  date="Yesterday"
                />
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}

function ActionCard({
  title,
  iconType,
}: {
  title: string;
  iconType: "phone" | "doc";
}) {
  return (
    <div className="w-48 h-40 bg-white border border-neutral-200 rounded-xl flex flex-col justify-center items-center gap-4 cursor-pointer transition hover:-translate-y-1 hover:shadow-md">
      {iconType === "phone" && (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          height="24px"
          viewBox="0 -960 960 960"
          width="24px"
          fill="#000000"
          className="w-12 h-12"
        >
          <path d="M480-800v-80h400v80H480Zm0 160v-80h400v80H480Zm0 160v-80h400v80H480ZM758-80q-125 0-247-54.5T289-289Q189-389 134.5-511T80-758q0-18 12-30t30-12h162q14 0 25 9.5t13 22.5l26 140q2 16-1 27t-11 19l-97 98q20 37 47.5 71.5T347-346q31 31 65 57.5t72 48.5l94-94q9-9 23.5-13.5T630-350l138 28q14 4 23 14.5t9 23.5v162q0 18-12 30t-30 12ZM201-560l66-66-17-94h-89q5 41 14 81t26 79Zm358 358q39 17 79.5 27t81.5 13v-88l-94-19-67 67ZM201-560Zm358 358Z" />{" "}
        </svg>
      )}

      {iconType === "doc" && (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          height="24px"
          viewBox="0 -960 960 960"
          width="24px"
          fill="#000000"
          className="w-12 h-12"
        >
          <path d="M320-440h320v-80H320v80Zm0 120h320v-80H320v80Zm0 120h200v-80H320v80ZM240-80q-33 0-56.5-23.5T160-160v-640q0-33 23.5-56.5T240-880h320l240 240v480q0 33-23.5 56.5T720-80H240Zm280-520v-200H240v640h480v-440H520ZM240-800v200-200 640-640Z" />{" "}
        </svg>
      )}
      <p className="font-medium">{title}</p>
    </div>
  );
}

function CallRow({
  name,
  phone,
  duration,
  status,
  date,
}: {
  name: string;
  phone: string;
  duration: string;
  status: string;
  date: string;
}) {
  return (
    <tr className="border-t border-neutral-200">
      <td className="px-6 py-4 font-medium">{name}</td>
      <td className="px-6 py-4 text-neutral-600">{phone}</td>
      <td className="px-6 py-4">{duration}</td>
      <td className="px-6 py-4">
        <span
          className={`px-3 py-1 rounded-full text-xs font-medium ${
            status === "Completed"
              ? "bg-green-100 text-green-700"
              : "bg-red-100 text-red-700"
          }`}
        >
          {status}
        </span>
      </td>
      <td className="px-6 py-4 text-neutral-500">{date}</td>
    </tr>
  );
}
