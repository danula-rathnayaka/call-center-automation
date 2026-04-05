"use client";

import { useMemo } from "react";
import { CallRow } from "../ui/CallRow";
import { ActionCard } from "../ui/ActionCard";
import { UserButton, useUser } from "@clerk/nextjs";
import { useRouter } from "next/navigation";

export default function AdminPage() {
  const { user } = useUser();
  const router = useRouter();

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
            <h1 className="text-2xl font-bold">
              Welcome back, {user?.firstName}
            </h1>
            <p className="text-sm text-neutral-500 mt-1">{today}</p>
          </div>

          <UserButton afterSignOutUrl="/" />
        </div>

        {/* Quick Action*/}
        <div className="mt-14">
          <h2 className="text-lg font-semibold mb-6">Quick Actions</h2>

          <div className="flex gap-6">
            <ActionCard title="Calls" iconType="phone" href="/admin/calls" />
            <ActionCard
              title="Documents"
              iconType="doc"
              href="/admin/documents"
            />
            <ActionCard title="Url" iconType="url" href="/admin/url" />
            <ActionCard title="Tools" iconType="api" href="/admin/tools" />
          </div>
        </div>

        {/* Recent Calls */}
        <div className="mt-16">
          <h2 className="text-lg font-semibold mb-6">Recent Calls</h2>

          <div className="bg-white rounded-xl border border-neutral-200 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-neutral-100 text-neutral-600">
                <tr>
                  <th className="text-left px-6 py-4">Phone</th>
                  <th className="text-left px-6 py-4">Duration</th>
                  <th className="text-left px-6 py-4">Status</th>
                  <th className="text-left px-6 py-4">Date</th>
                </tr>
              </thead>
              <tbody>
                <CallRow
                  phone="+94 77 123 4567"
                  duration="4m 12s"
                  status="Completed"
                  date="Today"
                />
                <CallRow
                  phone="+94 71 998 4455"
                  duration="—"
                  status="Missed"
                  date="Today"
                />
                <CallRow
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
