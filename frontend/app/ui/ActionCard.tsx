import Link from "next/link";

export function ActionCard({
  title,
  iconType,
  href,
}: {
  title: string;
  iconType: "phone" | "doc" | "api" | "url";
  href: string;
}) {
  return (
    <Link
      className="w-48 h-40 bg-white border border-neutral-200 rounded-xl flex flex-col justify-center items-center gap-4 cursor-pointer transition hover:-translate-y-1 hover:shadow-md"
      href={href}
    >
      {iconType === "phone" && (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          height="24px"
          viewBox="0 -960 960 960"
          width="24px"
          fill="#000000"
          className="w-12 h-12"
        >
          <path d="M480-800v-80h400v80H480Zm0 160v-80h400v80H480Zm0 160v-80h400v80H480ZM758-80q-125 0-247-54.5T289-289Q189-389 134.5-511T80-758q0-18 12-30t30-12h162q14 0 25 9.5t13 22.5l26 140q2 16-1 27t-11 19l-97 98q20 37 47.5 71.5T347-346q31 31 65 57.5t72 48.5l94-94q9-9 23.5-13.5T630-350l138 28q14 4 23 14.5t9 23.5v162q0 18-12 30t-30 12ZM201-560l66-66-17-94h-89q5 41 14 81t26 79Zm358 358q39 17 79.5 27t81.5 13v-88l-94-19-67 67ZM201-560Zm358 358Z" />{" "}
        </svg>
      )}

      {iconType === "doc" && (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          height="24px"
          viewBox="0 -960 960 960"
          width="24px"
          fill="#000000"
          className="w-12 h-12"
        >
          <path d="M320-440h320v-80H320v80Zm0 120h320v-80H320v80Zm0 120h200v-80H320v80ZM240-80q-33 0-56.5-23.5T160-160v-640q0-33 23.5-56.5T240-880h320l240 240v480q0 33-23.5 56.5T720-80H240Zm280-520v-200H240v640h480v-440H520ZM240-800v200-200 640-640Z" />{" "}
        </svg>
      )}

      {iconType === "api" && (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          height="24px"
          viewBox="0 -960 960 960"
          width="24px"
          fill="#000000"
          className="w-12 h-12"
        >
          <path d="m480-400-80-80 80-80 80 80-80 80Zm-85-235L295-735l185-185 185 185-100 100-85-85-85 85ZM225-295 40-480l185-185 100 100-85 85 85 85-100 100Zm510 0L635-395l85-85-85-85 100-100 185 185-185 185ZM480-40 295-225l100-100 85 85 85-85 100 100L480-40Z" />
        </svg>
      )}

      {iconType === "url" && (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          height="24px"
          viewBox="0 -960 960 960"
          width="24px"
          fill="#000000"
          className="w-12 h-12"
        >
          <path d="M440-280H280q-83 0-141.5-58.5T80-480q0-83 58.5-141.5T280-680h160v80H280q-50 0-85 35t-35 85q0 50 35 85t85 35h160v80ZM320-440v-80h320v80H320Zm200 160v-80h160q50 0 85-35t35-85q0-50-35-85t-85-35H520v-80h160q83 0 141.5 58.5T880-480q0 83-58.5 141.5T680-280H520Z" />
        </svg>
      )}
      <p className="font-medium">{title}</p>
    </Link>
  );
}
