import Link from "next/link";

export default function NotFound() {
  return (
    <main className="grid min-h-[100dvh] place-items-center bg-zinc-100 px-4 text-zinc-950">
      <div className="w-full max-w-sm rounded-md border border-zinc-200 bg-white p-6 text-center shadow-hairline">
        <p className="text-sm font-medium text-zinc-500">404</p>
        <h1 className="mt-2 text-xl font-semibold">Page not found</h1>
        <Link
          className="mt-5 inline-flex h-10 items-center rounded-md bg-zinc-950 px-4 text-sm font-medium text-white transition hover:bg-zinc-800"
          href="/"
        >
          Return home
        </Link>
      </div>
    </main>
  );
}
