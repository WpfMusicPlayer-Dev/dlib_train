// dlib_train.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#pragma execution_character_set("utf-8")


#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <dlib/svm_threaded.h>
#include <dlib/dnn.h>
#include <filesystem>

using line_sample_type = dlib::matrix<double, 0, 1>;
using song_sample_type = dlib::matrix<double>;

const int NUM_CLASSES = 8;
const int NUM_TYPES = 14;

using line_net = dlib::loss_multiclass_log<
    dlib::fc<NUM_CLASSES,
    dlib::relu<dlib::fc<64,
    dlib::relu<dlib::fc<128,
    dlib::input<line_sample_type>
    >>>>>>;

using song_net = dlib::loss_multiclass_log<
    dlib::fc<NUM_TYPES,
    dlib::relu<dlib::fc<32,
    dlib::relu<dlib::fc<64,
    dlib::input<song_sample_type>
    >>>>>>;

struct TrainingRow {
    std::string label;
    std::string text;
};

static const std::unordered_map<std::string, unsigned long> line_table = {
    {"zh", 0},
    {"jp", 1},
    {"kr", 2},
    {"latin", 3}, // en, fr, itai, es, etc -> mapped to latin
    {"ru", 4}, // add russian support.
    {"jyut", 5},
    {"roma", 6},
    {"onomatopoeia", 7}
};

static const std::unordered_map<std::string, unsigned long> song_table = {
    {"zh_only", 0},
    {"jp_only", 1},
    {"kr_only", 2},
    {"latin_only", 3},
    {"ru_only", 4},
    {"jp_zh_trans", 5},
    {"jp_roma", 6},
    {"latin_zh_trans", 7},
    {"ru_zh_trans", 8},
    {"kr_zh_trans", 9},
    {"kr_roma", 10},
    {"zh_jyut", 11},
    {"jp_zh_trans_roma", 12},
    {"kr_zh_trans_roma", 13}
};

void build_vocab(const std::vector<TrainingRow>& data, std::unordered_map<std::string, int>& vocab)
{
    int index = 0;
    for (const auto& row : data)
    {
        const std::string& text = row.text;
        for (size_t i = 0; i < text.size(); ++i)
        {
            for (int n = 1; n <= 3; ++n)
            {
                if (i + n > text.size()) continue;
                std::string gram = text.substr(i, n);
                if (!std::all_of(gram.begin(), gram.end(), [](unsigned char c) { return c != '\t' && c != '\r' && c != '\n'; }))
                    continue;
                if (vocab.find(gram) == vocab.end())
                    vocab[gram] = index++;
            }
        }
    }
}

std::vector<double> extract_features(const std::string& text, const std::unordered_map<std::string, int>& vocab)
{
    std::vector<double> x(vocab.size(), 0.0);

    for (size_t i = 0; i < text.size(); ++i)
    {
        for (int n = 1; n <= 3; ++n)
        {
            if (i + n > text.size()) continue;
            std::string gram = text.substr(i, n);
            auto it = vocab.find(gram);
            if (it != vocab.end())
                x[it->second] += 1.0;
        }
    }
    return x;
}
inline unsigned long label_to_id(const std::string& label)
{
    auto it = line_table.find(label);
    if (it != line_table.end())
        return it->second;

    return 6;
}

inline std::string id_to_label(unsigned long id) 
{
    auto it = std::find_if(line_table.begin(), line_table.end(), [id](const std::pair<std::string, unsigned long>& key) -> bool {
        return key.second == id;
    });
    if (it != line_table.end())
        return it->first;

    return "unknown";
}


std::vector<TrainingRow> load_training_data(const std::string& filename)
{
    std::vector<TrainingRow> rows;
    if (!std::filesystem::exists(filename))
        throw std::runtime_error("file not exists");
    std::ifstream fin(filename);
    if (!fin.is_open()) return rows;

    std::string line;
    while (std::getline(fin, line))
    {
        if (line.empty()) continue;

        size_t tab = line.find(' ');
        if (tab == std::string::npos) continue;

        TrainingRow row;
        row.label = line.substr(0, tab);
        row.text = line.substr(tab + 1);

        rows.push_back(row);
    }
    return rows;
}

void line_train() 
{
    std::unordered_map<std::string, int> vocab;
    std::vector<line_sample_type> samples;
    std::vector<unsigned long> labels;
    auto training_data = load_training_data("train.tsv");

    // 构建 vocab
    build_vocab(training_data, vocab);
    if (vocab.empty()) {
        std::cerr << "vocab is empty\n";
        return;
    }

    // 构造训练数据
    for (auto& row : training_data)
    {
        auto feat = extract_features(row.text, vocab);
        line_sample_type m(feat.size());
        for (size_t i = 0; i < feat.size(); ++i)
            m(i) = feat[i];

        samples.push_back(m);
        labels.push_back(label_to_id(row.label));
    }

    // 训练
    line_net net;
    dlib::dnn_trainer<line_net> trainer(net);
    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.0001);
    trainer.set_iterations_without_progress_threshold(200);
    trainer.be_verbose();

    trainer.train(samples, labels);

    // 保存模型
    dlib::serialize("lyric_lang_mlp.dat") << net << vocab;
}

void line_reasoning() 
{
    // 推理测试
    line_net net_reasoning;
    std::unordered_map<std::string, int> vocab_reasoning;

    dlib::deserialize("lyric_lang_mlp.dat") >> net_reasoning >> vocab_reasoning;

    std::initializer_list<std::string> test_cases = {
        "우리 다시 돌아갈 수는 없을까",
        "lang yu ye o bed soeng gwei ga ",
        "Words are not enough action speaks louder",
        "ki mi no mo to ni ho ra to do i te ho shi i da ke ",
        "得到佢睇起你",
        "Oh oh la la la",
        "Bang Bang Bang! Bang Bang Bang!",
        "to u ri a me ko ko ro mo yo u se tsu na no de a i",
        "kyo ku: Aiobahn",
        "啊啊",
        "zu tto",
        "Quand il me prend dans ses bras",
        "Lorem ipsum dolor sit amet",
        "Симпа па па полюбила",
        "Парней она манила",
        "Бродягу полюбила",
        "Она его половина"
    };
    for (const auto& test_case : test_cases) {
        auto feat = extract_features(test_case, vocab_reasoning);
        line_sample_type m(feat.size());
        for (size_t i = 0; i < feat.size(); ++i)
            m(i) = feat[i];

        int label_id = net_reasoning(m);
        printf("test_case = %s, label id = %s\n", test_case.c_str(), id_to_label(label_id).c_str());
    }
}

// 构建特征向量
dlib::matrix<double> extract_song_features(const std::vector<std::string>& seq)
{
    const int LANGS = 8; // zh jp kr latin jyut roma ono
    song_sample_type feat(84, 1); 
    feat = 0;

    // 语言计数
    std::vector<int> count(LANGS, 0);
    for (auto& s : seq)
    {
        if (s == "zh") count[0]++;
        else if (s == "jp") count[1]++;
        else if (s == "kr") count[2]++;
        else if (s == "latin") count[3]++;
        else if (s == "ru") count[4]++;
        else if (s == "jyut") count[5]++;
        else if (s == "roma") count[6]++;
        else count[7]++; // onomatopoeia
    }

    int idx = 0;

    // 语言计数（8维）
    for (int i = 0; i < LANGS; i++)
        feat(idx++) = count[i];

    // 语言比例（8维）
    double total = seq.size();
    for (int i = 0; i < LANGS; i++)
        feat(idx++) = count[i] / total;

    // bigram矩阵（64维）
    std::vector<std::vector<int>> trans(LANGS, std::vector<int>(LANGS, 0));
    for (size_t i = 1; i < seq.size(); i++)
    {
        auto prev = seq[i - 1];
        auto curr = seq[i];

        auto id = [&](const std::string& s) {
            if (s == "zh") return 0;
            if (s == "jp") return 1;
            if (s == "kr") return 2;
            if (s == "latin") return 3;
            if (s == "ru") return 4;
            if (s == "jyut") return 5;
            if (s == "roma") return 6;
            return 7;
            };

        trans[id(prev)][id(curr)]++;
    }

    for (int i = 0; i < LANGS; i++)
        for (int j = 0; j < LANGS; j++)
            feat(idx++) = trans[i][j];

    // 语言切换次数（1维）
    int switches = 0;
    for (size_t i = 1; i < seq.size(); i++)
        if (seq[i] != seq[i - 1])
            switches++;
    feat(idx++) = switches;

    // 是否包含翻译（1维）
    feat(idx++) = (count[0] > 0 && (count[1] > 0 || count[2] > 0 || count[3] > 0));

    // 是否包含罗马音（1维）
    feat(idx++) = (count[5] > 0);

    // 是否包含粤拼（1维）
    feat(idx++) = (count[4] > 0);

    return feat;
}

void song_train()
{
    std::vector<song_sample_type> samples;
    std::vector<unsigned long> labels;

    std::ifstream fin("song_train.tsv");
    if (!fin.is_open()) {
        std::cerr << "song_train.tsv not found\n";
        return;
    }

    std::string line;
    while (std::getline(fin, line))
    {
        if (line.empty()) continue;

        size_t tab = line.find(' ');
        if (tab == std::string::npos) continue;

        std::string label = line.substr(0, tab);
        std::string seq_str = line.substr(tab + 1);

        std::stringstream ss(seq_str);
        std::string token;
        std::vector<std::string> seq;
        while (ss >> token)
            seq.push_back(token);

        auto feat = extract_song_features(seq);

        samples.push_back(feat);
        labels.push_back(song_table.at(label));
    }

    song_net net;
    dlib::dnn_trainer<song_net> trainer(net);
    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.0001);
    trainer.set_iterations_without_progress_threshold(200);
    trainer.be_verbose();

    trainer.train(samples, labels);

    dlib::serialize("song_structure_mlp.dat") << net;
}

void song_reasoning()
{
    song_net net;
    dlib::deserialize("song_structure_mlp.dat") >> net;

    std::vector<std::string> seq = { "latin", "zh", "latin", "zh", "onomatopoeia", "zh", "onomatopoeia", "onomatopoeia", "latin", "zh" };

    auto feat = extract_song_features(seq);
    int id = net(feat);

    printf("song type id = %d\n", id);
}

int main()
{
    system("chcp 65001");
    setlocale(LC_ALL, "zh-CN.UTF-8");
    SetConsoleOutputCP(65001);
    line_reasoning();
    // song_train();
    // song_reasoning();
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
