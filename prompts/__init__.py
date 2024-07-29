from typing import Callable, List

from . import alfworld
from . import coa, hotpotQA, fever, webshop, travelplanner
from .templates.human import *
from .templates.system import *

FEWSHOTS = dict(
    coa=coa.FEWSHOTS,
    hotpotqa=hotpotQA.FEWSHOTS,
    fever=fever.FEWSHOTS,
    alfworld=alfworld.FEWSHOTS,
    webshop=webshop.FEWSHOTS,
)
REFLECTION_FEWSHOTS = dict(
    coa=coa.REFLECTION_FEWSHOTS,
    hotpotqa=hotpotQA.REFLECTION_FEWSHOTS,
    fever=None,#fever.REFLECTION_FEWSHOTS,
    alfworld=alfworld.REFLECTION_FEWSHOTS,
    webshop=webshop.REFLECTION_FEWSHOTS,
)
SYSTEM_INSTRUCTION = dict(
    coa=coa.SYSTEM_INSTRUCTION,
    hotpotqa=hotpotQA.SYSTEM_INSTRUCTION,
    fever=fever.SYSTEM_INSTRUCTION,
    alfworld=alfworld.SYSTEM_INSTRUCTION,
    webshop=webshop.SYSTEM_INSTRUCTION,
)
SYSTEM_REFLECTION_INSTRUCTION = dict(
    coa=coa.SYSTEM_REFLECTION_INSTRUCTION,
    hotpotqa=hotpotQA.SYSTEM_REFLECTION_INSTRUCTION,
    fever=None,#fever.SYSTEM_REFLECTION_INSTRUCTION,
    alfworld=None,#alfworld.SYSTEM_REFLECTION_INSTRUCTION,
    webshop=None,#webshop.SYSTEM_REFLECTION_INSTRUCTION,
)
HUMAN_INSTRUCTION = dict(
    coa=coa.HUMAN_INSTRUCTION,
    hotpotqa=hotpotQA.HUMAN_INSTRUCTION,
    fever=fever.HUMAN_INSTRUCTION,
    alfworld=alfworld.HUMAN_INSTRUCTION,
    webshop=webshop.HUMAN_INSTRUCTION,
)
HUMAN_REFLECTION_INSTRUCTION = dict(
    coa=coa.HUMAN_REFLECTION_INSTRUCTION,
    hotpotqa=hotpotQA.HUMAN_REFLECTION_INSTRUCTION,
    fever=None,
    alfworld=alfworld.HUMAN_REFLECTION_INSTRUCTION,
    webshop=webshop.HUMAN_REFLECTION_INSTRUCTION,
)
SYSTEM_CRITIQUE_INSTRUCTION = dict(
    coa=dict(
        compare_existing_rules=coa.SYSTEM_CRITIQUE_EXISTING_RULES_INSTRUCTION,
        all_success_existing_rules=coa.SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_RULES_INSTRUCTION,
    ),
    hotpotqa=dict(
        compare_existing_rules=hotpotQA.SYSTEM_CRITIQUE_EXISTING_RULES_INSTRUCTION,
        all_success_existing_rules=hotpotQA.SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_RULES_INSTRUCTION,
    ),
    fever=dict(
        compare_existing_rules=None,
        all_success_existing_riles=None
    ),
    alfworld=dict(
        compare_existing_rules=alfworld.SYSTEM_CRITIQUE_EXISTING_RULES_INSTRUCTION,
        all_success_existing_rules=alfworld.SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_RULES_INSTRUCTION,
    ),
    webshop=dict(
        compare_existing_rules=webshop.SYSTEM_CRITIQUE_EXISTING_RULES_INSTRUCTION,
        all_success_existing_rules=webshop.SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_RULES_INSTRUCTION,
    ),
)

LLM_PARSER = dict(
    coa=coa.LLM_PARSER,
    hotpotqa=hotpotQA.LLM_PARSER,
    # fever and hotpotQA has same format
    fever=hotpotQA.LLM_PARSER,
    alfworld=alfworld.LLM_PARSER,
    webshop=webshop.LLM_PARSER,
)

OBSERVATION_FORMATTER = dict(
    coa=coa.OBSERVATION_FORMATTER,
    hotpotqa=hotpotQA.OBSERVATION_FORMATTER,
    # fever and hotpotQA has same format
    fever=hotpotQA.OBSERVATION_FORMATTER,
    alfworld=alfworld.OBSERVATION_FORMATTER,
    webshop=webshop.OBSERVATION_FORMATTER,
)

STEP_IDENTIFIER = dict(
    coa=coa.STEP_IDENTIFIER,
    hotpotqa=hotpotQA.STEP_IDENTIFIER,
    # fever and hotpotQA has same format
    fever=hotpotQA.STEP_IDENTIFIER,
    webshop=webshop.STEP_IDENTIFIER,
    alfworld=alfworld.STEP_IDENTIFIER,
)

CYCLER = dict(
    coa=coa.CYCLER,
    hotpotqa=hotpotQA.CYCLER,
    fever=hotpotQA.CYCLER,
    # fever and hotpotQA has same format
    webshop=webshop.CYCLER,
    alfworld=alfworld.CYCLER,
)
REFLECTION_PREFIX = dict(
    coa=coa.REFLECTION_PREFIX,
    hotpotqa=hotpotQA.REFLECTION_PREFIX,
    fever=hotpotQA.REFLECTION_PREFIX,
    alfworld=alfworld.REFLECTION_PREFIX,
    # same format as alfworld
    webshop=webshop.REFLECTION_PREFIX,
)
PREVIOUS_TRIALS_FORMATTER=dict(
    coa=coa.PREVIOUS_TRIALS_FORMATTER,
    hotpotqa=hotpotQA.PREVIOUS_TRIALS_FORMATTER,
    fever=hotpotQA.PREVIOUS_TRIALS_FORMATTER,
    alfworld=alfworld.PREVIOUS_TRIALS_FORMATTER,
    # same format as alfworld
    webshop=webshop.PREVIOUS_TRIALS_FORMATTER,
)

STEP_STRIPPER = dict(
    coa=coa.STEP_STRIPPER,
    hotpotqa=hotpotQA.STEP_STRIPPER,
    fever=hotpotQA.STEP_STRIPPER,
    alfworld=alfworld.STEP_STRIPPER,
    # same format as alfworld
    webshop=webshop.STEP_STRIPPER,
)

def STEP_CYCLER(benchmark: str, lines: str, cycler: Callable, step_identifier: Callable, stripper: Callable = lambda x, y: x) -> List[str]:
    steps = []
    scratch_pad = ''
    for line in cycler(lines):
        step_type = step_identifier(line)
        stripped_line = stripper(line, step_type)
        scratch_pad += stripped_line + '\n'
        if step_type == 'observation':
            steps.append(scratch_pad.strip())
            scratch_pad = ''
    if scratch_pad != '':
        steps.append(scratch_pad.strip())
    return steps
