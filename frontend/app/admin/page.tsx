"use client";

import { useMemo, useState, useEffect } from "react";
import { CallRow } from "../ui/CallRow";
import { ActionCard } from "../ui/ActionCard";
import { UserButton, useUser } from "@clerk/nextjs";
import { useRouter } from "next/navigation";

export default function AdminPage() {
  const { user } = useUser();
  const router = useRouter();
  const [recentCalls, setRecentCalls] = useState<any[]>([]);

  const fetchQueue = async () => {
    try {
      // History
      const historyRes = await fetch(
        "http://localhost:8000/api/handoff/history",
      );
      const historyData = await historyRes.json();

      setRecentCalls(historyData.items || []);
    } catch (err) {
      console.error("Failed to fetch data:", err);
    }
  };

  useEffect(() => {
    fetchQueue();

    const interval = setInterval(fetchQueue, 5000);
    return () => clearInterval(interval);
  }, []);

  const today = useMemo(() => {
    return new Date().toLocaleDateString("en-US", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  }, []);

  const formatDuration = (start: string, end: string) => {
    if (!start || !end) return "—";

    const diff = Math.floor(
      (new Date(end).getTime() - new Date(start).getTime()) / 1000,
    );

    const mins = Math.floor(diff / 60);
    const secs = diff % 60;

    return `${mins}m ${secs}s`;
  };

  const formatDateLabel = (dateStr: string) => {
    const date = new Date(dateStr);
    const today = new Date();
    const yesterday = new Date();
    yesterday.setDate(today.getDate() - 1);

    if (date.toDateString() === today.toDateString()) return "Today";
    if (date.toDateString() === yesterday.toDateString()) return "Yesterday";

    return date.toLocaleDateString();
  };

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
        <div className="mt-3">
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
                {recentCalls.length > 0 ? (
                  recentCalls.map((call) => {
                    const duration = formatDuration(
                      call.answered_at,
                      call.actioned_at,
                    );

                    const status =
                      call.status === "ended"
                        ? "Completed"
                        : call.answered_at
                          ? "Completed"
                          : "Missed";

                    return (
                      <CallRow
                        key={call.id}
                        phone={call.phone_number}
                        duration={duration}
                        status={status}
                        date={formatDateLabel(call.created_at)}
                      />
                    );
                  })
                ) : (
                  <tr>
                    <td
                      colSpan={4}
                      className="text-center py-10 text-neutral-400"
                    >
                      No recent calls available
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}
