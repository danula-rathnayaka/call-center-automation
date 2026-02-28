"use client";

import { CallRow } from "@/app/ui/CallRow";
import { useRouter } from "next/navigation";

export default function AdminCalls() {
  const router = useRouter();

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
            <div className="w-full rounded-xl bg-red-100 py-4 px-4 flex justify-between items-center">
              <div>+94 71 998 4455</div>
              <div className="w-10 h-10 bg-green-700 flex items-center justify-center rounded-full cursor-pointer">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  height="24px"
                  viewBox="0 -960 960 960"
                  width="24px"
                  fill="#FFFFFF"
                  className=""
                >
                  <path d="M798-120q-125 0-247-54.5T329-329Q229-429 174.5-551T120-798q0-18 12-30t30-12h162q14 0 25 9.5t13 22.5l26 140q2 16-1 27t-11 19l-97 98q20 37 47.5 71.5T387-386q31 31 65 57.5t72 48.5l94-94q9-9 23.5-13.5T670-390l138 28q14 4 23 14.5t9 23.5v162q0 18-12 30t-30 12ZM241-600l66-66-17-94h-89q5 41 14 81t26 79Zm358 358q39 17 79.5 27t81.5 13v-88l-94-19-67 67ZM241-600Zm358 358Z" />
                </svg>
              </div>
            </div>
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
