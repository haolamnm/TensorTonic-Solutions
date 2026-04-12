def schedule_pipeline(tasks, resource_budget):
    schedule = []       # (task_name, start_time)
    done = set()        # names of completed tasks
    running = {}        # name -> end_time

    tm = 0

    while len(done) < len(tasks):

        # complete all tasks that have finished by current time
        for name, end_time in list(running.items()):
            if tm >= end_time:
                done.add(name)
                del running[name]

        # current resource usage from still-running tasks
        curr_resources = sum(
            task["resources"]
            for task in tasks
            if task["name"] in running
        )

        # start any task that is ready (deps met, not done/running, fits budget)
        for task in tasks:
            if (task["name"] not in done
                and task["name"] not in running
                and all(dep in done for dep in task["depends_on"])
                and curr_resources + task["resources"] <= resource_budget
            ):
                schedule.append((task["name"], tm))
                running[task["name"]] = tm + task["duration"]
                curr_resources += task["resources"]

        # advance time to the next task completion
        if running:
            tm = min(running.values())
        elif len(done) < len(tasks):
            break  # nothing running and tasks remain — unsatisfiable

    return schedule