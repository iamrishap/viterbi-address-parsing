import re

State_File = './toy_example/State_File'
Symbol_File = './toy_example/Symbol_File'
Query_File = './toy_example/Query_File'

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File):
    trans_prob = create_transition_matrix(State_File)
    evidence = create_emission_matrix(Symbol_File)
    observations = open(Query_File).readlines()
    states = [str(i) for i in range(int(open(State_File).readline()))]

    # Setting the start prob of all the items as 0 expect the BEGIN. BEGIN is kept as 1.
    start_prob = {}
    for i in range(len(states)):
        if i == len(states) - 2:
            start_prob[str(i)] = 1
        else:
            start_prob[str(i)] = 0

    for observation in observations:
        raw_observation_array = parse_query_tokens(observation.rstrip())
        cleaned_observation_array = get_cleaned_observations(raw_observation_array, Symbol_File)
        # print(cleaned_observation_array)
        print(compute_path_using_viterbi(cleaned_observation_array, states, start_prob, trans_prob, evidence))


# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k):
    pass


# Question 3
def advanced_decoding(State_File, Symbol_File, Query_File):
    pass


def parse_query_tokens(address_line):
    '''
    Splits an address passed based on various split separators.
    :param address_line: Address to be parsed.
        Example Address: 8/23-35 Barker St., Kingsford, NSW 2032
    :return: Array of parsed tokens.
        Example Parsed:
        ['8', '/', '23', '-', '35', 'Barker', 'St.', ',', 'Kingsford', ',', 'NSW', '2032']
    '''
    tokens_with_spaces = re.split('([ *,/()\\-&])', address_line)  #
    tokens = [token for token in tokens_with_spaces if token not in ['', ' ']]
    return tokens


def create_transition_matrix(state_file_path):
    state_file_handler = open(state_file_path)
    transition_matrix = {}
    number_of_states = int(state_file_handler.readline())
    for i in range(number_of_states):
        state_file_handler.readline()  # Just consuming the state lines. State names are useless.
        transition_matrix[str(i)] = {'total_obs': 0}

    transitions = state_file_handler.readlines()
    for transition in transitions:
        start_state, end_state, frequency = transition.split()
        transition_matrix[start_state][end_state] = int(frequency)
        transition_matrix[start_state]['total_obs'] += int(frequency)

    # Calculating the probability (with smoothing)
    for i in range(number_of_states):
        state_freq_total = transition_matrix[str(i)]['total_obs']
        for key, value in transition_matrix[str(i)].items():
            if key != 'total_obs':
                transition_matrix[str(i)][key] = (value + 1) / (state_freq_total + number_of_states - 1)
        del(transition_matrix[str(i)]['total_obs'])

        # Setting the frequency as 0 for the missing values
        for j in range(number_of_states):
            if str(j) not in transition_matrix[str(i)]:
                transition_matrix[str(i)][str(j)] = 1 / (state_freq_total + number_of_states - 1)

    state_file_handler.close()
    return transition_matrix


def create_emission_matrix(symbol_file_path):
    symbol_file_handler = open(symbol_file_path)
    emission_matrix = {}
    number_of_states = int(symbol_file_handler.readline())
    symbols_dict = {}
    for i in range(number_of_states):
        symbol_name = symbol_file_handler.readline().rstrip()
        symbols_dict[str(i)] = symbol_name
        emission_matrix[str(i)] = {'total_obs': 0}

    transitions = symbol_file_handler.readlines()
    for transition in transitions:
        state, emission, frequency = transition.split()
        emission_matrix[state][symbols_dict[emission]] = int(frequency)
        emission_matrix[state]['total_obs'] += int(frequency)

    for i in range(number_of_states):
        emission_freq_total = emission_matrix[str(i)]['total_obs']
        for key, value in emission_matrix[str(i)].items():
            if key != 'total_obs':
                emission_matrix[str(i)][key] = (value + 1) / (emission_freq_total + number_of_states + 1)

        # Setting the frequency as 0 for the missing values
        for value in symbols_dict.values():
            if value not in emission_matrix[str(i)]:
                emission_matrix[str(i)][value] = 1 / (emission_freq_total + number_of_states + 1)

        # Setting the probability for the unknown
        emission_matrix[str(i)]['UNK'] = 1 / (emission_freq_total + number_of_states + 1)

        del (emission_matrix[str(i)]['total_obs'])

    # Adding the frequency for the BEGIN and END state
    emission_matrix[str(number_of_states)] = {}
    emission_matrix[str(number_of_states + 1)] = {}
    for value in symbols_dict.values():
        emission_matrix[str(number_of_states)][value] = 1 / (number_of_states + 1)
        emission_matrix[str(number_of_states + 1)][value] = 1 / (number_of_states + 1)

    emission_matrix[str(number_of_states)]['UNK'] = 1 / (number_of_states + 1)
    emission_matrix[str(number_of_states + 1)]['UNK'] = 1 / (number_of_states + 1)
    symbol_file_handler.close()
    return emission_matrix


def compute_path_using_viterbi(observations, states, start_prob, trans_prob, evidence):
    path = {state: [] for state in states}  # Creating a dictionary and initialising it.
    cur_prob = {}  # Initializing current probability to empty dictionary.
    for state in states:
        cur_prob[state] = start_prob[state] * evidence[state][observations[0]]  # Calculate the current probability values.

    # print cur_prob[state]
    for i in range(1, len(observations)):
        last_prob = cur_prob
        cur_prob = {}
        for curr_state in states:
            # below is the recurrence relation to compute max_prob
            maxProb, last_state = max(((last_prob[last_state] * trans_prob[last_state][curr_state] *
                                        evidence[curr_state][observations[i]], last_state)
                                       for last_state in states))
            cur_prob[curr_state] = maxProb
            path[curr_state].append(last_state)

    # Find the maximum Probability value.
    maxProb = -1
    maxPath = None
    # In each state find the max prob and return that.
    for state in states:
        path[state].append(state)
        if cur_prob[state] > maxProb:
            # print maxPath
            maxPath = path[state]
            maxProb = cur_prob[state]
        # print maxProb
    return maxPath


def get_cleaned_observations(raw_observation_array, symbol_file_path):
    valid_symbols = set()
    symbol_file_handler = open(symbol_file_path)
    number_of_symbols = int(symbol_file_handler.readline())

    for i in range(number_of_symbols):
        valid_symbols.add(symbol_file_handler.readline().rstrip())

    cleaned_observation_array = [
                                    observation
                                    if observation in valid_symbols
                                    else 'UNK'
                                    for observation in raw_observation_array
                                 ]
    return cleaned_observation_array

viterbi_algorithm(State_File, Symbol_File, Query_File)
# print(create_transition_matrix(State_File))
# print(create_emission_matrix(Symbol_File))

# Missing probabilies should be zero. Ex. 3 in state matrix 0 has 3 and 3 has 3,4 4 has 0
# Missing in 3, 4 00