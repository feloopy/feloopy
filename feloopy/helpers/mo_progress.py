import sys


class MOProgress:
    """Live-updating multi-objective progress dashboard.

    Works for any MO method (NWSM, ECM, etc.). Shows:
    - Header with method name, objectives, directions, intervals
    - Payoff table (optional)
    - Step-by-step progress with flexible details
    - Final result with Pareto points and conflict metric
    """

    def __init__(self, method, M, directions, intervals):
        from rich.console import Console
        from rich.live import Live
        
        self.method = method
        self.M = M
        self.directions = directions
        self.intervals = intervals
        self.rows = []
        self.payoff = None
        self.console = Console()
        self.live = Live(self._build_renderable(), console=self.console,
                         refresh_per_second=8, transient=False)
        self._started = False

    def start(self):
        self.live.start()
        self._started = True

    def stop(self):
        if self._started:
            self.live.stop()
            self._started = False

    def _build_renderable(self):
        # imported here (not in __init__) because these are free names in this
        # method; __init__'s locals are invisible to it.
        from rich import box
        from rich.console import Group
        from rich.table import Table

        dir_str = ", ".join(
            f"{'↑' if d == 'max' else '↓'} obj{i+1}"
            for i, d in enumerate(self.directions)
        )

        header = Table(
            box=box.ROUNDED, show_header=False, show_edge=True,
            title=f"[bold]Multi-Objective ({self.method.upper()})[/bold]",
            title_style="bold cyan", border_style="cyan", padding=(0, 1),
        )
        header.add_column("Key", style="dim")
        header.add_column("Value")
        header.add_row("Objectives", str(self.M))
        header.add_row("Directions", dir_str)
        header.add_row("Intervals", str(self.intervals))

        parts = [header]

        if self.payoff is not None:
            payoff_tbl = Table(
                box=box.SIMPLE_HEAVY, show_header=False, show_edge=True,
                title="[bold]Payoff Table[/bold]", title_style="bold yellow",
                border_style="yellow", padding=(0, 1),
            )
            for i in range(self.M):
                row_str = "  ".join(
                    f"{self.payoff[i, j]:8.2f}" for j in range(self.M)
                )
                payoff_tbl.add_row(f"[{row_str}]")
            parts.append(payoff_tbl)

        steps_tbl = Table(
            box=box.SIMPLE, show_header=False, show_edge=True,
            title=f"[bold]{self.method.upper()} Progress[/bold]",
            title_style="bold green", border_style="green", padding=(0, 1),
        )
        steps_tbl.add_column("Step", style="dim", width=12)
        steps_tbl.add_column("Details")
        steps_tbl.add_column("Result", style="bold")
        for label, detail, result in self.rows:
            steps_tbl.add_row(label, detail, result)
        parts.append(steps_tbl)

        return Group(*parts)

    def update(self):
        self.live.update(self._build_renderable())

    def add_step(self, label, detail, result):
        self.rows.append((label, detail, result))

    def set_payoff(self, payoff):
        self.payoff = payoff

    def finish(self, pareto_count, conflict):
        self.add_step(
            "[bold green]Done[/bold green]",
            f"Pareto points: [bold]{pareto_count}[/bold]",
            f"Conflict: [bold]{conflict:.4f}[/bold]",
        )
        self.update()


class NullMOProgress:
    """No-op progress when verbose=False."""

    def __init__(self, *a, **kw): pass
    def start(self): pass
    def stop(self): pass
    def update(self): pass
    def add_step(self, *a): pass
    def set_payoff(self, *a): pass
    def finish(self, *a): pass
