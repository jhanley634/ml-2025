// Scrapes and formats DOM elements from NYT Connections results.
//
// Read the DOM by pasting this into a browser console.

{
    const labeledElements = document.querySelectorAll("[aria-label]");
    const results = [];
    labeledElements.forEach((element) => {
        const label = element.getAttribute("aria-label");
        if (label && label.startsWith("Correct group ")) {
            const parts = label.split(". ");
            const groupName = parts[0].replace("Correct group ", "").trim();
            const items = parts
                .slice(1)
                .join(". ")
                .split(",")
                .map((item) => item.trim());
            const formattedItems = items.join(", ");
            results.push(`| ${groupName} | ${formattedItems} |`);
        }
    });
    console.log(results.join("\n"));
}
