from collections.abc import Callable
from typing import Any

from botocore.exceptions import ClientError

Invocation = dict[str, Any]


class FakeSsm:
    """Stands in for the SSM client: moto does not implement instance information
    and runs no commands. `on_command(script)` returns the invocation polls to serve;
    parameters are read from `params` (a moto SSM client)."""

    def __init__(self, on_command: Callable[[str], list[Invocation]] | None = None) -> None:
        self.on_command = on_command or (lambda script: [{"Status": "Success"}])
        self.sent: list[dict[str, Any]] = []
        self.scripts: list[str] = []
        self.cancelled: list[str] = []
        self._polls: dict[str, list[Invocation]] = {}
        self.online = True
        self.params: Any = None

    def send_command(self, **kwargs: Any) -> dict[str, Any]:
        self.sent.append(kwargs)
        commands = kwargs["Parameters"]["commands"]
        script = commands[1].split("\n", 1)[1].rsplit("\nLOOM_SCRIPT_EOF", 1)[0]
        self.scripts.append(script)
        command_id = f"cmd-{len(self.sent)}"
        self._polls[command_id] = [
            {"Status": "InvocationDoesNotExist"},
            *self.on_command(script),
        ]
        return {"Command": {"CommandId": command_id}}

    def get_command_invocation(self, CommandId: str, InstanceId: str) -> Invocation:
        polls = self._polls[CommandId]
        inv = polls.pop(0) if len(polls) > 1 else polls[0]
        if inv["Status"] == "InvocationDoesNotExist":
            raise ClientError(
                {"Error": {"Code": "InvocationDoesNotExist", "Message": "no"}},
                "GetCommandInvocation",
            )
        return inv

    def cancel_command(self, CommandId: str, InstanceIds: list[str]) -> dict[str, Any]:
        self.cancelled.append(CommandId)
        return {}

    def describe_instance_information(self, Filters: list[dict[str, Any]]) -> dict[str, Any]:
        ids = Filters[0]["Values"]
        status = "Online" if self.online else "ConnectionLost"
        return {"InstanceInformationList": [{"InstanceId": i, "PingStatus": status} for i in ids]}

    def get_parameter(self, Name: str) -> dict[str, Any]:
        return dict(self.params.get_parameter(Name=Name))


class Proxy:
    """Wraps a real (moto) client, overriding selected methods."""

    def __init__(self, inner: Any, **overrides: Callable[..., Any]) -> None:
        self._inner = inner
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._inner, name)


def client_error(code: str, op: str = "RunInstances") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, op)


async def no_sleep(_: float) -> None:
    return None
