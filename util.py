from reserved_words import reserved
import string
def read_data(filenames, is_test=False):
    link_stack = []
    cluster_stat = {}
    start_count = 0
    isolate_count = 0
    instances = []
    link_stat = {}
    a= 0
    b=0
    f = open("self_link_start_of_the conversation.txt",'w')
    f_1 = open("self_link_isolated_non_system_message.txt",'w')
    for filename in filenames:
        name = filename
        for ending in [".annotation.txt", ".ascii.txt", ".raw.txt", ".tok.txt"]:
            if filename.endswith(ending):
                name = filename[:-len(ending)]
        link_stat[name+".annotation.txt"] = {}
        # cluster stores all the conversation inside a channel
        cluster_stat[name+".annotation.txt"] = {}
        text_ascii = [l.strip().split() for l in open(name +".ascii.txt")]
        text = [l.strip()for l in open(name +".ascii.txt")]
        text_tok = []
        for l in open(name +".tok.txt"):
            l = l.strip().split()
            if len(l) > 0 and l[-1] == "</s>":
                l = l[:-1]
            if len(l) == 0 or l[0] != '<s>':
                l.insert(0, "<s>")
            text_tok.append(l)
        info, target_info = lines_to_info(text_ascii)
        links = {}
        if is_test:
            for i in range(args.test_start, min(args.test_end, len(text_ascii))):
                links[i] = []
        else:
            for line in open(name +".annotation.txt"):
                nums = [int(v) for v in line.strip().split() if v != '-']
                head = min(nums)
                tail = max(nums)
                if not head in cluster_stat[name+".annotation.txt"] :
                    cluster_stat[name+".annotation.txt"][head] = len(cluster_stat[name+".annotation.txt"]) +1
                cluster_stat[name+".annotation.txt"][tail] = cluster_stat[name+".annotation.txt"][head]
                if head in link_stack:
                    link_stack.remove(head)
                    link_stat[name+".annotation.txt"][head] = 'conversation_start'
                    f.write(text[head]+'\n')
                    start_count +=1
                if tail == head:
                    link_stack.append(tail)
                links.setdefault(max(nums), []).append(min(nums))
            isolate_count += len(link_stack)
            for query in link_stack:
                if "===" not in text[query]:
                    f_1.write(text[query]+'\n')
                    link_stat[name+".annotation.txt"][query] = 'nonsystem_self_link'
                else:
                    link_stat[name+".annotation.txt"][query] = 'system_self_link'
            link_stack = []
        for link, nums in links.items():
            if link in link_stat[name+".annotation.txt"]:
                stat  = link_stat[name+".annotation.txt"][link]
                a +=1
            else:
                link_stat[name+".annotation.txt"][link] = 'nonself_link'
                stat= 'nonself_link'
                b +=1
            instances.append((name +".annotation.txt", link, nums, text_ascii, text_tok, info, target_info,stat))
    print("#isolate_count:%d" % (isolate_count))
    print("#start_conv_count:%d" % (start_count))
    print("#",a,b)
    return instances,link_stat,cluster_stat
FEATURES = 77
###############################################################################
def update_user(users, user):
    if user in reserved:
        return
    all_digit = True
    for char in user:
        if char not in string.digits:
            all_digit = False
    if all_digit:
        return
    users.add(user.lower())
def update_users(line, users):
    if len(line) < 2:
        return
    user = line[1]
    if user in ["Topic", "Signoff", "Signon", "Total", "#ubuntu"
            "Window", "Server:", "Screen:", "Geometry", "CO,",
            "Current", "Query", "Prompt:", "Second", "Split",
            "Logging", "Logfile", "Notification", "Hold", "Window",
            "Lastlog", "Notify", 'netjoined:']:
        # Ignore as these are channel commands
        pass
    else:
        if line[0].endswith("==="):
            parts = ' '.join(line).split("is now known as")
            if len(parts) == 2 and line[-1] == parts[-1].strip():
                user = line[-1]
        elif line[0][-1] == ']':
            if user[0] == '<':
                user = user[1:]
            if user[-1] == '>':
                user = user[:-1]
        user = user.lower()
        update_user(users, user)
        # This is for cases like a user named |blah| who is
        # refered to as simply blah
        core = [char for char in user]
        while len(core) > 0 and core[0] in string.punctuation:
            core.pop(0)
        while len(core) > 0 and core[-1] in string.punctuation:
            core.pop()
        core = ''.join(core)
        update_user(users, core)
# Names two letters or less that occur more than 500 times in the data
common_short_names = {"ng", "_2", "x_", "rq", "\\9", "ww", "nn", "bc", "te",
"io", "v7", "dm", "m0", "d1", "mr", "x3", "nm", "nu", "jc", "wy", "pa", "mn",
"a_", "xz", "qr", "s1", "jo", "sw", "em", "jn", "cj", "j_"}
def get_targets(line, users):
    targets = set()
    for token in line[2:]:
        token = token.lower()
        user = None
        if token in users and len(token) > 2:
            user = token
        else:
            core = [char for char in token]
            while len(core) > 0 and core[-1] in string.punctuation:
                core.pop()
                nword = ''.join(core)
                if nword in users and (len(core) > 2 or nword in common_short_names):
                    user = nword
                    break
            if user is None:
                while len(core) > 0 and core[0] in string.punctuation:
                    core.pop(0)
                    nword = ''.join(core)
                    if nword in users and (len(core) > 2 or nword in common_short_names):
                        user = nword
                        break
        if user is not None:
            targets.add(user)
    return targets
def lines_to_info(text_ascii):
    users = set()
    for line in text_ascii:
        update_users(line, users)
    chour = 12
    cmin = 0
    info = []
    target_info = {}
    nexts = {}
    for line_no, line in enumerate(text_ascii):
        if line[0].startswith("["):
            user = line[1][1:-1]
            nexts.setdefault(user, []).append(line_no)
    prev = {}
    for line_no, line in enumerate(text_ascii):
        user = line[1]
        system = True
        if line[0].startswith("["):
            chour = int(line[0][1:3])
            cmin = int(line[0][4:6])
            user = user[1:-1]
            system = False
        is_bot = (user == 'ubottu' or user == 'ubotu')
        targets = get_targets(line, users)
        for target in targets:
            target_info.setdefault((user, target), []).append(line_no)
        last_from_user = prev.get(user, None)
        if not system:
            prev[user] = line_no
        next_from_user = None
        if user in nexts:
            while len(nexts[user]) > 0 and nexts[user][0] <= line_no:
                nexts[user].pop(0)
            if len(nexts[user]) > 0:
                next_from_user = nexts[user][0]
        info.append((user, targets, chour, cmin, system, is_bot, last_from_user, line, next_from_user))
    return info, target_info
def get_time_diff(info, a, b):
    if a is None or b is None:
        return -1
    if a > b:
        t = a
        a = b
        b = t
    ahour = info[a][2]
    amin = info[a][3]
    bhour = info[b][2]
    bmin = info[b][3]
    if ahour == bhour:
        return bmin - amin
    if bhour < ahour:
        bhour += 24
    return (60 - amin) + bmin + 60*(bhour - ahour - 1)
cache = {}