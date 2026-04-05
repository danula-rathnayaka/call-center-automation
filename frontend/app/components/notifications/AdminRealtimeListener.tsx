"use client";

import { useEffect } from "react";
import { useNotification } from "./NotificationProvider";
import Pusher from "pusher-js";

export function AdminRealtimeListener() {
  const { notify } = useNotification();

  useEffect(() => {
    const pusher = new Pusher(process.env.NEXT_PUBLIC_PUSHER_KEY!, {
      cluster: "ap2",
    });

    const channel = pusher.subscribe("admin-channel");

    channel.bind("escalation", (data: any) => {
      notify({
        title: "Escalation Alert",
        message: data.message,
        type: "warning",
        duration: 10,
      });
    });

    return () => {
      pusher.unsubscribe("admin-channel");
      pusher.disconnect();
    };
  }, [notify]);

  return null;
}
