import { signOutAction } from "@/app/actions";

export function SignOutButton({ className = "text-sm text-zinc-600 hover:text-zinc-900" }: { className?: string }) {
  return (
    <form action={signOutAction}>
      <button type="submit" className={className}>
        Sign out
      </button>
    </form>
  );
}
