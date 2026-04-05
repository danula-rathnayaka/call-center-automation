"use client";

import { useNotification } from "@/app/components/notifications/NotificationProvider";
import { CallRow } from "@/app/ui/CallRow";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

export default function AdminCalls() {
  const router = useRouter();
  const { notify } = useNotification();

  const [priorityCalls, setPriorityCalls] = useState<any[]>([]);
  const [answeredCalls, setAnsweredCalls] = useState<any[]>([]);
  const [recentCalls, setRecentCalls] = useState<any[]>([]);

  const fetchQueue = async () => {
    try {
      // Queue
      const res = await fetch("http://localhost:8000/api/handoff/queue");
      const data = await res.json();

      setPriorityCalls(
        data.items.filter((item: any) => item.status === "ringing"),
      );

      setAnsweredCalls(
        data.items.filter((item: any) => item.status === "answered"),
      );

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

  const handleCloseCall = async (id: string) => {
    try {
      await fetch(`http://localhost:8000/api/handoff/${id}/end`, {
        method: "POST",
      });

      notify({
        title: "Call Closed",
        message: "The call has been successfully closed",
        type: "success",
      });

      fetchQueue();
    } catch (err) {
      console.error(err);
    }
  };

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
      <main className="flex-1 p-10 border-x border-neutral-200">
        {/* Header */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => router.back()}
            className="p-2 bg-neutral-100 hover:bg-neutral-200 transition rounded-full cursor-pointer"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="24px"
              viewBox="0 -960 960 960"
              width="24px"
              fill="#000000"
            >
              <path d="M560-240 320-480l240-240 56 56-184 184 184 184-56 56Z" />
            </svg>
          </button>
          <div>
            <h1 className="text-2xl font-bold">Call Management</h1>
          </div>
        </div>

        {/* Priority Calls */}
        <div className="mt-3">
          <h2 className="text-lg font-semibold mb-6">Priority Calls</h2>

          <div className="flex gap-4">
            {priorityCalls.map((call) => (
              <div
                key={call.id}
                className="w-full rounded-xl bg-red-100 py-4 px-4 flex justify-between items-center"
              >
                <div>{call.phone_number}</div>
                <button
                  onClick={() =>
                    notify({
                      title: "Escalation Alert",
                      message: call.escalation_reason || call.query,
                      type: "info",
                    })
                  }
                  className="w-10 h-10 bg-green-700 flex items-center justify-center rounded-full cursor-pointer"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    height="24px"
                    viewBox="0 -960 960 960"
                    width="24px"
                    fill="#FFFFFF"
                  >
                    <path d="M798-120q-125 0-247-54.5T329-329Q229-429 174.5-551T120-798q0-18 12-30t30-12h162q14 0 25 9.5t13 22.5l26 140q2 16-1 27t-11 19l-97 98q20 37 47.5 71.5T387-386q31 31 65 57.5t72 48.5l94-94q9-9 23.5-13.5T670-390l138 28q14 4 23 14.5t9 23.5v162q0 18-12 30t-30 12Z" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Answered Calls */}
        <div className="mt-6">
          <h2 className="text-lg font-semibold mb-6">Answered Calls</h2>

          <div className="flex gap-4">
            {answeredCalls.map((call) => (
              <div
                key={call.id}
                className="w-full rounded-xl bg-green-100 py-4 px-4 flex justify-between items-center"
              >
                <div>{call.phone_number}</div>

                <button
                  onClick={() => handleCloseCall(call.id)}
                  className="w-10 h-10 bg-red-600 flex items-center justify-center rounded-full cursor-pointer"
                >
                  {/* Close Icon */}
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    height="20px"
                    viewBox="0 -960 960 960"
                    width="20px"
                    fill="#FFFFFF"
                  >
                    <path d="M256-200 200-256l224-224-224-224 56-56 224 224 224-224 56 56-224 224 224 224-56 56-224-224-224 224Z" />
                  </svg>
                </button>
              </div>
            ))}
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
