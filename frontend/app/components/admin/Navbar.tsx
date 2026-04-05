"use client";
import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";

export function NavBar() {
  const [isSession, setSession] = useState<boolean>(true);
  const router = useRouter();

  useEffect(() => {
    if (!isSession) {
      router.push("/login");
    }
  }, [isSession, router]);

  return (
    <div className="fixed top-0 left-0 w-full z-50 flex justify-between items-center py-4 px-6 bg-[#001337] shadow-lg">
      <div className="text-white font-bold text-2xl">Call Center</div>

      <div className="flex items-center">
        {isSession && (
          <button className="cursor-pointer">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="24px"
              viewBox="0 -960 960 960"
              width="24px"
              fill="#FFFFFF"
            >
              <path d="M200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h280v80H200v560h280v80H200Zm440-160-55-58 102-102H360v-80h327L585-622l55-58 200 200-200 200Z" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}
