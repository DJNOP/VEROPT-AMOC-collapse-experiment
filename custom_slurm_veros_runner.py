# custom_slurm_veros_runner.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from veropt.interfaces.slurm_simulation import (
    SlurmVerosRunner,
    SlurmSimulation,
    try_to_run,
    write_batch_script_string,
    create_batch_script,
)
from veropt.interfaces.veros_utility import edit_veros_run_script as _edit_settings_only


class CustomSlurmVerosRunner(SlurmVerosRunner):
    """SlurmVerosRunner that can overwrite non-`settings.*` parameters in setup scripts.

    Default runner only overwrites lines like:
        settings.<key> = ...

    We additionally overwrite a literal constant line in the setup file:
        SSS_OFFSET = 0.0

    So you can open point_X/global_4_degree.py and *see* the value used.
    """

    def set_up_and_run(
        self,
        simulation_id: str,
        parameters: dict[str, float],
        run_script_directory: str,
        run_script_filename: str,
        output_filename: str,
    ):
        run_script_file = os.path.join(run_script_directory, f"{run_script_filename}.py")
        batch_script_filename = f"veros_batch_{simulation_id}"
        batch_script_file = os.path.join(run_script_directory, f"{batch_script_filename}.sh")
        slurm_log_filename = f"slurm_{simulation_id}"
        slurm_log_file = os.path.join(run_script_directory, f"{slurm_log_filename}.out")

        if not self.config.keep_old_params:
            # 1) keep default behaviour (settings.<key> overwrites + prints)
            _edit_settings_only(run_script=run_script_file, parameters=parameters)

            # 2) add our custom overwrites
            self._edit_custom_parameters(run_script=run_script_file, parameters=parameters)

        substitutions_dict = {
            "simulation_id": simulation_id,
            "run_script_filename": run_script_filename,
            "batch_script_filename": batch_script_filename,
            "slurm_log_filename": slurm_log_filename,
            "output_filename": output_filename,
        }

        template_substitutions = self.config.model_dump() | substitutions_dict

        batch_script_string = write_batch_script_string(
            batch_script_template=self.config.batch_script_template,
            template_substitutions=template_substitutions,
        )

        create_batch_script(
            batch_script_file=batch_script_file,
            batch_script_string=batch_script_string,
        )

        simulation = SlurmSimulation(
            simulation_id=simulation_id,
            run_script_directory=run_script_directory,
            output_filename=output_filename,
            batch_script_file=batch_script_file,
        )

        result = try_to_run(
            simulation=simulation,
            parameters=parameters,
            max_tries=self.config.max_tries,
        )

        result.slurm_log_file = slurm_log_file
        return result

    def _edit_custom_parameters(self, run_script: str, parameters: dict[str, Any]) -> None:
        if "sss_offset" in parameters:
            self._overwrite_constant(
                run_script=run_script,
                constant_name="SSS_OFFSET",
                value=float(parameters["sss_offset"]),
                pretty_key="sss_offset",
            )

    @staticmethod
    def _overwrite_constant(
        run_script: str,
        constant_name: str,
        value: float,
        pretty_key: str | None = None,
    ) -> None:
        key_for_print = pretty_key or constant_name
        pattern = re.compile(rf"^(?P<indent>\s*){re.escape(constant_name)}\s*=\s*.*$")

        p = Path(run_script)
        lines = p.read_text().splitlines(True)

        new_lines: list[str] = []
        replaced = False

        for line in lines:
            m = pattern.match(line)
            if m and not replaced:
                indent = m.group("indent")
                old_assignment = line.strip()
                new_line = (
                    f"{indent}{constant_name} = {value}  "
                    f"# default: {old_assignment}\n"
                )
                print(f"Overwriting {key_for_print} with value: {value}")
                new_lines.append(new_line)
                replaced = True
            else:
                new_lines.append(line)

        if not replaced:
            print(
                f"WARNING: {key_for_print} was provided but no line like "
                f"'{constant_name} = ...' was found to overwrite in {run_script}."
            )

        p.write_text("".join(new_lines))
