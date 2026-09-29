"""A second implementation of one campaign, written from the equations in the paper (Section 4), not from the
vectorised code. It loops over agents one at a time and uses only the standard library for arithmetic. It draws its
random numbers in the same order as the main implementation (one uniform per agent per day, then one Poisson draw
for deaths), so on a shared seed the two must agree exactly.

It covers the stylized reference specification: fixed bandh days, Poisson deaths, normal thresholds, no concession.
"""
import math

GROUP_COUNT = 4
SC, ST, OBC, GENERAL = 0, 1, 2, 3


def material_change(agent, parameters):
    group, above_line, most_deprived = agent["group"], agent["above_line"], agent["most_deprived"]
    if group in (SC, ST):
        m = parameters.material_loss_sc_st_above_income_line if above_line else parameters.material_loss_sc_st_below_income_line
        if most_deprived:
            m -= parameters.sub_classification_gain_for_most_deprived_tier
    elif group == OBC:
        m = 0.0 if above_line else parameters.material_loss_obc_below_income_line
    else:
        m = 0.0 if above_line else parameters.material_loss_general_below_income_line
    return parameters.loss_aversion * m if m > 0 else m


def symbolic_threat(agent, parameters):
    tier_share = parameters.most_deprived_tier_share_of_symbolic_threat if agent["most_deprived"] else 1.0
    return float(parameters.symbolic_threat_by_group[agent["group"]]) * tier_share


def simulate_reference_campaign(population, parameters, random_generator):
    agents = [
        {"group": int(population.social_group[i]), "above_line": bool(population.is_above_income_line[i]),
         "most_deprived": bool(population.is_most_deprived_tier[i]), "identity": float(population.identity_strength[i]),
         "neighbourhood": int(population.neighbourhood[i]),
         "threshold": parameters.mean_participation_threshold + parameters.participation_threshold_spread * float(population.threshold_standard_score[i])}
        for i in range(population.agent_count)
    ]
    for agent in agents:
        agent["material"] = material_change(agent, parameters)
        agent["symbolic"] = symbolic_threat(agent, parameters)

    beta_n, beta_v = parameters.max_neighbourhood_influence, parameters.max_national_visibility_influence
    bandh_mobilization, kappa = parameters.mobilization_on_bandh_days, parameters.deaths_per_crore_protester_days
    if parameters.internet_shutdown_active:
        beta_n *= parameters.internet_shutdown_coordination_factor
        beta_v *= parameters.internet_shutdown_coordination_factor
        bandh_mobilization *= parameters.internet_shutdown_bandh_mobilization_factor
        kappa *= parameters.internet_shutdown_violence_factor

    neighbourhood_members = {}
    for index, agent in enumerate(agents):
        neighbourhood_members.setdefault(agent["neighbourhood"], []).append(index)
    group_sizes = [sum(1 for agent in agents if agent["group"] == g) for g in range(GROUP_COUNT)]
    people_per_agent = 146e7 / len(agents)

    neighbourhood_share = {c: 0.0 for c in neighbourhood_members}
    group_share = [0.0] * GROUP_COUNT
    martyr_rise = [0.0] * GROUP_COUNT
    ever_protested = [False] * len(agents)
    daily_protesters, daily_by_group, daily_deaths = [], [], []

    for day in range(parameters.campaign_length_days):
        is_bandh = day in parameters.bandh_call_days
        b_t = bandh_mobilization if is_bandh else parameters.mobilization_on_ordinary_days
        probabilities = []
        for agent in agents:
            g = agent["group"]
            symbolic_now = agent["symbolic"] + (martyr_rise[g] if agent["symbolic"] > 0 else 0.0)
            grievance = parameters.material_loss_weight * agent["material"] + parameters.symbolic_threat_weight * agent["identity"] * symbolic_now
            pull = (beta_n * math.tanh(neighbourhood_share[agent["neighbourhood"]] / parameters.neighbourhood_turnout_at_saturation)
                    + beta_v * math.tanh(group_share[g] / parameters.national_turnout_at_saturation))
            push = float(parameters.organizational_capacity_by_group[g]) * parameters.opposition_party_amplifier * b_t
            net = grievance + pull + push - agent["threshold"]
            if is_bandh:
                net -= parameters.heavy_policing_turnout_cost
            probabilities.append(1.0 / (1.0 + math.exp(-net / parameters.decision_noise)))
        uniforms = random_generator.random(len(agents))
        protests = [float(uniforms[i]) < probabilities[i] for i in range(len(agents))]

        group_counts = [0] * GROUP_COUNT
        for index, agent in enumerate(agents):
            if protests[index]:
                ever_protested[index] = True
                agent["threshold"] += parameters.fatigue_per_protest_day
                group_counts[agent["group"]] += 1
        for c, members in neighbourhood_members.items():
            neighbourhood_share[c] = sum(1 for i in members if protests[i]) / len(members)
        group_share = [group_counts[g] / group_sizes[g] for g in range(GROUP_COUNT)]
        protesters_today = sum(group_counts) * people_per_agent

        deaths = int(random_generator.poisson(kappa * protesters_today / 1e7))
        if deaths:
            total = max(sum(group_counts), 1)
            for g in range(GROUP_COUNT):
                martyr_rise[g] = min(martyr_rise[g] + parameters.symbolic_threat_rise_per_death * deaths * group_counts[g] / total * GROUP_COUNT,
                                     parameters.max_martyrdom_symbolic_rise)
        daily_protesters.append(protesters_today)
        daily_by_group.append([count * people_per_agent for count in group_counts])
        daily_deaths.append(deaths)

    return {"daily_protesters": daily_protesters, "daily_by_group": daily_by_group, "daily_deaths": daily_deaths,
            "peak": max(daily_protesters), "cumulative": sum(ever_protested) * people_per_agent, "deaths": sum(daily_deaths)}
