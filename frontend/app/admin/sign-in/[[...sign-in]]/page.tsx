"use client";

import { SignIn } from "@clerk/nextjs";

export default function Page() {
  return (
    <>
      <div className="flex min-h-screen bg-neutral-50 text-neutral-800">
        <main className="flex-1 flex justify-center items-center p-10 border-x border-neutral-200">
          <SignIn signUpUrl="/admin/sign-up" />
        </main>
      </div>
    </>
  );
}
