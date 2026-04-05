"use client";

import CallDetailsDialog from "@/app/components/admin/CallDetailsDialog";
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
  const [isCallDialogOpen, setIsCallDialogOpen] = useState(false);
  const [selectedCall, setSelectedCall] = useState<any | null>(null);

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

  const handleOpenCallDetails = (call: any) => {
    setSelectedCall(call);
    setIsCallDialogOpen(true);
  };

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

  const handleAnswerCall = async (call: any) => {
    try {
      await fetch(`http://localhost:8000/api/handoff/${call.id}/answer`, {
        method: "POST",
      });
      setSelectedCall(call);
      setIsCallDialogOpen(true);
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
    <>
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

            <div className="flex  flex-col gap-4 w-full">
              {priorityCalls.length > 0 ? (
                priorityCalls.map((call) => (
                  <div
                    key={call.id}
                    className="w-full rounded-xl bg-red-100 py-4 px-4 flex justify-between items-center"
                  >
                    <div>{call.phone_number}</div>
                    <button
                      onClick={() => handleAnswerCall(call)}
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
                ))
              ) : (
                <div className="w-full text-center py-8 text-neutral-400 bg-red-50 rounded-xl">
                  No priority calls right now
                </div>
              )}
            </div>
          </div>

          {/* Answered Calls */}
          <div className="mt-6">
            <h2 className="text-lg font-semibold mb-6">Answered Calls</h2>

            <div className="flex flex-col gap-4 w-full">
              {answeredCalls.length > 0 ? (
                answeredCalls.map((call) => (
                  <div
                    key={call.id}
                    className="w-full rounded-xl bg-green-100 py-4 px-4 flex justify-between items-center"
                  >
                    <div>{call.phone_number}</div>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                      <span className="text-xs text-green-700">
                        Active call
                      </span>
                    </div>
                    <button
                      onClick={() => handleOpenCallDetails(call)}
                      className="w-10 h-10 bg-blue-600 hover:bg-blue-700 flex items-center justify-center rounded-full cursor-pointer transition"
                      title="Open Call Details"
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        height="24px"
                        viewBox="0 -960 960 960"
                        width="24px"
                        fill="#ffffff"
                      >
                        <path d="M607.5-372.5Q660-425 660-500t-52.5-127.5Q555-680 480-680t-127.5 52.5Q300-575 300-500t52.5 127.5Q405-320 480-320t127.5-52.5Zm-204-51Q372-455 372-500t31.5-76.5Q435-608 480-608t76.5 31.5Q588-545 588-500t-31.5 76.5Q525-392 480-392t-76.5-31.5ZM214-281.5Q94-363 40-500q54-137 174-218.5T480-800q146 0 266 81.5T920-500q-54 137-174 218.5T480-200q-146 0-266-81.5ZM480-500Zm207.5 160.5Q782-399 832-500q-50-101-144.5-160.5T480-720q-113 0-207.5 59.5T128-500q50 101 144.5 160.5T480-280q113 0 207.5-59.5Z" />
                      </svg>
                    </button>
                  </div>
                ))
              ) : (
                <div className="w-full text-center py-8 text-neutral-400 bg-green-50 rounded-xl">
                  No active answered calls
                </div>
              )}
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
      <CallDetailsDialog
        isOpen={isCallDialogOpen}
        onClose={() => setIsCallDialogOpen(false)}
        call={selectedCall}
        onEnd={handleCloseCall}
      />
    </>
  );
}
