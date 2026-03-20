import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const { message } = await req.json();

  let reply = "";

  if (message.toLowerCase().includes("balance")) {
    reply = "Your account balance is twelve thousand five hundred rupees.";
  } else if (message.toLowerCase().includes("agent")) {
    reply = "Connecting you to a human agent.";
  } else {
    reply = "I understand. How can I assist you further?";
  }

  return NextResponse.json({
    reply,
  });
}
