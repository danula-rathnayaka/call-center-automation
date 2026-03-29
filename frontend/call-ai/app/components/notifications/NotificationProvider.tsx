"use client";

import { createContext, useContext, useState, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle, XCircle, Info, AlertTriangle } from "lucide-react";

type NotificationType = "success" | "error" | "info" | "warning";

type Notification = {
  id: string;
  title?: string;
  message: string;
  type?: NotificationType;
  duration?: number;
};

type NotificationContextType = {
  notify: (n: Omit<Notification, "id">) => void;
};

const NotificationContext = createContext<NotificationContextType | null>(null);

export const useNotification = () => {
  const ctx = useContext(NotificationContext);
  if (!ctx)
    throw new Error("useNotification must be used inside NotificationProvider");
  return ctx;
};

export function NotificationProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const notify = useCallback((n: Omit<Notification, "id">) => {
    const id = crypto.randomUUID();
    const newNote: Notification = {
      id,
      duration: n.duration ?? 4000,
      ...n,
    };

    setNotifications((prev) => [...prev, newNote]);

    setTimeout(() => {
      setNotifications((prev) => prev.filter((x) => x.id !== id));
    }, newNote.duration);
  }, []);

  const iconMap: Record<NotificationType, React.ReactNode> = {
    success: <CheckCircle size={22} className="text-green-500" />,
    error: <XCircle size={22} className="text-red-500" />,
    info: <Info size={22} className="text-blue-500" />,
    warning: <AlertTriangle size={22} className="text-yellow-500" />,
  };

  return (
    <NotificationContext.Provider value={{ notify }}>
      {children}

      {/* Notification Container */}
      <div className="fixed top-5 right-5 flex flex-col gap-3 z-[9999]">
        <AnimatePresence>
          {notifications.map((n) => {
            const Icon = n.type ? iconMap[n.type] : iconMap["info"]; // fallback icon

            return (
              <motion.div
                key={n.id}
                initial={{ opacity: 0, y: -20, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -20, scale: 0.98 }}
                transition={{ duration: 0.25, ease: "easeOut" }}
                className="
                  rounded-2xl shadow-xl bg-white/80 backdrop-blur-md
                  border border-neutral-200 px-4 py-3 w-80
                  flex items-start gap-3
                "
              >
                {/* ICON */}
                <div className="pt-1">{Icon}</div>

                {/* TEXT */}
                <div className="flex-1">
                  {n.title && (
                    <p className="font-semibold text-neutral-900">{n.title}</p>
                  )}
                  <p className="text-sm text-neutral-800">{n.message}</p>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </NotificationContext.Provider>
  );
}
