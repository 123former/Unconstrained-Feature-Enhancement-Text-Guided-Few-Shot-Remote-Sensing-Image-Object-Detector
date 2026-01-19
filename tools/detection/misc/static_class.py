import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter

# 配置路径
jpeg_folder = '/home/f523/disk1/sxp/mmfewshot/data/DIOR/JPEGImages-test'
anno_folder = '/home/f523/disk1/sxp/mmfewshot/data/DIOR/Annotations'
output_plot = '/home/f523/disk1/sxp/mmfewshot/data/DIOR/sentence_count_frequency.png'  # 保存的图像文件名

# 类别与文本描述的关系
category_to_description = {
    "airplane": "An airplane typically has symmetrical wings distributed on both sides of the fuselage.",
    "baseballfield": "A baseball field is usually a quarter-circle diamond shape.",
    "basketballcourt": "A basketball court typically features a rectangular floor.",
    "bridge": "A bridge is generally a slender structure spanning obstacles.",
    "groundtrackfield": "A ground track field usually contains an oval-shaped running track.",
    "harbor": "A harbor is typically a sheltered, curved body of water.",
    "ship": "A ship is typically a long, narrow hull with pointed ends.",
    "storagetank": "A storage tank is usually vertical cylindrical.",
    "tenniscourt": "A tennis court typically has a rectangular surface.",
    "vehicle": "A vehicle is often streamlined with four wheels.",
    "airport": "An airport usually features rectangular runways and terminal buildings.",
    "chimney": "A chimney is commonly conical, large at the base and narrow at the top.",
    "dam": "A dam is generally narrow at the top and wide at the bottom.",
    "Expressway-Service-area": "An expressway service area typically includes a linear arrangement of rectangular buildings and parking areas.",
    "Expressway-toll-station": "An expressway toll station is characterized by adjacent toll booths spanning the highway.",
    "golffield": "A golf field is often a wide area with intertwined fairways.",
    "overpass": "An overpass is generally an elevated structure crossing road, often curved or straight.",
    "stadium": "A stadium is commonly round or oval in shape.",
    "trainstation": "A train station generally consists of multiple rectangular platforms and linear tracks.",
    "windmill": "A windmill typically has three blades mounted on a tall central column."
}

category_to_description2 = {
    "airplane": "An airplane includes fuselage, wings, tail, and engines, etc. The two wings of the aircraft are symmetrically distributed on both sides of the fuselage.",
    "airport": "An airport consists of a landing area with airport aprons and runways for planes to take off and to land. It includes adjacent utility buildings such as control towers, hangars and terminals.",
    "dam": "Dams are usually built along the water edge, narrow at the top and wide at the bottom. They are barriers that prevent or restrict the flow of surface water or underground streams.",
    "Expressway-Service-area": "An expressway service area is usually located on both sides of the highway. It includes parking areas and buildings for people to rest and shop.",
    "Expressway-toll-station": "An expressway toll station is an enclosure placed along a toll road and consists of several adjacent tell booths and spans the entire highway.",
    "golffield": "A golf field is halfway up the mountain and usually covers a wide area with many fairways intertwined.",
    "groundtrackfield": "A ground track field contains the outer oval shaped running track and an area of turf within this track.",
    "harbor": "A harbor is a sheltered body of water. It provides docking areas for ships, boats, and barges.",
    "overpass": "An overpass is a bridge or road that crosses over another road or railway.",
    "stadium": "Stadium is round or oval. It contains auditoriums, a running track, and a football pitch.",
    "storagetank": "Storage tanks are usually vertical cylindrical. They are white or gray.",
    "tenniscourt": "It is a firm rectangular surface with a low net stretched across the center.",
    "trainstation": "A train station generally consists of multiple platforms, multiple tracks and a station building.",
    "vehicle": "The vehicle contains four tires and has windshields front and rear. Vehicles drive on the road or are stationed in the parking lot.",
    "windmill": "A windmill is a structure that contains three blades which are erected by a tall column.",
    "baseballfield": "A baseball diamond is a quarter circle. It is covered by grass.",
    "basketballcourt": "The basketball court consists of a rectangular floor with baskets at each end. Outdoor surfaces are generally made from standard paving materials.",
    "bridge": "A bridge is usually slender. It is a structure built to span a physical obstacle such as a body of water, valley, road, or rail.",
    "chimney": "Chimneys are usually conical in shape, large below and small above. They are typically vertical, or as close to vertical as possible.",
    "ship": "Ships are usually found in oceans, rivers, lakes, etc. where there is water. The shape of a ship is a long bar with pointed ends."
}

category_to_description3 = {
    "airplane": "An airplane includes fuselage, wings, tail, and engines, etc. The two wings of the aircraft are symmetrically distributed on both sides of the fuselage. It is typically adjacent to runways at airports, surrounded by taxiways and terminal buildings.",
    "airport": "An airport consists of a landing area with airport aprons and runways for planes to take off and to land. It includes adjacent utility buildings such as control towers, hangars and terminals. It is usually connected to highways and surrounded by service areas or industrial zones.",
    "dam": "Dams are usually built along the water edge, narrow at the top and wide at the bottom. They are barriers that prevent or restrict the flow of surface water or underground streams. They are located between mountains or valleys, with reservoirs upstream and river channels downstream.",
    "Expressway-Service-area": "An expressway service area is usually located on both sides of the highway. It includes parking areas and buildings for people to rest and shop. It is often adjacent to gas stations and toll stations, with vegetation buffers separating it from the main road.",
    "Expressway-toll-station": "An expressway toll station is an enclosure placed along a toll road and consists of several adjacent tell booths and spans the entire highway. It is typically preceded by speed reduction markings and followed by service areas or highway exits.",
    "golffield": "A golf field is halfway up the mountain and usually covers a wide area with many fairways intertwined. It is surrounded by natural terrain features like forests or water hazards, with clubhouses located at the periphery.",
    "groundtrackfield": "A ground track field contains the outer oval shaped running track and an area of turf within this track. It is commonly adjacent to soccer fields or sports stadiums, with spectator stands along one side.",
    "harbor": "A harbor is a sheltered body of water. It provides docking areas for ships, boats, and barges. It is surrounded by container terminals, cranes, and breakwaters, with ships anchored in designated zones.",
    "overpass": "An overpass is a bridge or road that crosses over another road or railway. It is flanked by on/off ramps and connected to ground-level intersections or highways.",
    "stadium": "Stadium is round or oval. It contains auditoriums, a running track, and a football pitch. It is often surrounded by parking lots, commercial facilities, and public transportation hubs.",
    "storagetank": "Storage tanks are usually vertical cylindrical. They are white or gray. They are clustered in industrial zones, spaced apart for safety and connected by pipelines and access roads.",
    "tenniscourt": "It is a firm rectangular surface with a low net stretched across the center. It is often grouped with other courts in sports complexes, surrounded by fences and spectator seating.",
    "trainstation": "A train station generally consists of multiple platforms, multiple tracks and a station building. It is connected to urban transit systems like buses or subways, with parking facilities nearby.",
    "vehicle": "The vehicle contains four tires and has windshields front and rear. Vehicles drive on the road or are stationed in the parking lot. It is commonly found on roads adjacent to buildings, gas stations, or residential areas.",
    "windmill": "A windmill is a structure that contains three blades which are erected by a tall column. It is arranged in rows on open plains or hills, spaced apart to maximize wind exposure.",
    "baseballfield": "A baseball diamond is a quarter circle. It is covered by grass. It is often located in parks or school campuses, surrounded by outfield fences and bleachers.",
    "basketballcourt": "The basketball court consists of a rectangular floor with baskets at each end. Outdoor surfaces are generally made from standard paving materials. It is commonly found in schoolyards, parks, or urban recreational areas with adjacent benches or playgrounds.",
    "bridge": "A bridge is usually slender. It is a structure built to span a physical obstacle such as a body of water, valley, road, or rail. It is connected to approach roads on both ends and often features guardrails or lighting systems.",
    "chimney": "Chimneys are usually conical in shape, large below and small above. They are typically vertical, or as close to vertical as possible. They are located on industrial plants or power stations, often grouped with other smokestacks.",
    "ship": "Ships are usually found in oceans, rivers, lakes, etc. where there is water. The shape of a ship is a long bar with pointed ends. It is moored at docks alongside container terminals or anchored in open water near harbors."
}

# 计算每张图片上文本描述的句子数量
def count_sentences_per_image(image_names):
    sentence_counts = []

    for img_name in image_names:
        xml_file = os.path.join(anno_folder, img_name + '.xml')

        if not os.path.exists(xml_file):
            print(f"Warning: Annotation file {xml_file} not found. Skipping.")
            continue

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            sentences = set()
            for obj in root.findall('object'):
                name_elem = obj.find('name').text.strip()
                description = category_to_description3.get(name_elem, "")
                num_sentences = len(description.split('.'))
                sentences.add(num_sentences)

            sentence_counts.append(sum(sentences))

        except ET.ParseError as e:
            print(f"Error parsing {xml_file}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue

    return sentence_counts


# 获取所有图片的名字（不包括扩展名）
image_files = [f for f in os.listdir(jpeg_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]
image_names = [os.path.splitext(f)[0] for f in image_files]

print(f"Found {len(image_names)} images in {jpeg_folder}")

# 统计句子数量
sentence_counts = count_sentences_per_image(image_names)

if not sentence_counts:
    print("No valid annotations found. Cannot plot.")
else:
    # Count frequency of each unique sentence count
    count_freq = Counter(sentence_counts)
    counts = sorted(count_freq.keys())
    freqs = [count_freq[c] for c in counts]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(counts, freqs, color='skyblue', edgecolor='black', width=0.8)
    plt.xlabel('Number of Sentences per Image')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sentence Counts per Image')
    plt.xticks(counts)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_plot, dpi=200)
    print(f"Bar chart saved as: {output_plot}")

    # Optionally display the plot (if running locally)
    # plt.show()

    # Print statistics
    print(f"Processed {len(sentence_counts)} images with annotations.")
    print(f"Average number of sentences per image: {sum(sentence_counts) / len(sentence_counts):.2f}")
    print(f"Maximum sentences in one image: {max(sentence_counts)}")
    print(f"Minimum sentences in one image: {min(sentence_counts)}")