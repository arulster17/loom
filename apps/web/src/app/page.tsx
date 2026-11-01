import { redirect } from "next/navigation";

import { auth } from "@/auth";

export default async function Home() {
  redirect((await auth())?.user ? "/dashboard" : "/signin");
}
