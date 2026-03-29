import { pusherServer } from "@/lib/pusher";
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const { query } = await req.json();

  let reply = "";

  if (query.toLowerCase().includes("balance")) {
    reply = "Your account balance is twelve thousand five hundred rupees.";
  } else if (query.toLowerCase().includes("agent")) {
    reply = "Connecting you to a human agent.";
  } else {
    reply = "I understand. How can I assist you further?";
  }

  const escalate = true;

  if (escalate) {
    await pusherServer.trigger("admin-channel", "escalation", {
      message: "A call needs admin attention",
    });
  }

  return NextResponse.json({
    response: reply,
    escalate: true,
  });
}
