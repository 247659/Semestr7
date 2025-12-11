package bdd.pages.contact;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

public class MessagesPage {

    private final WebDriver driver;
    private final WebDriverWait wait;

    private final By tableRows = By.cssSelector("table.table-hover");

    public MessagesPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, java.time.Duration.ofSeconds(10));
    }

    public void open() {
        driver.get("http://localhost:4200/account/messages");
        waitUntilTableIsLoaded();
    }

    public void openAsAdmin() {
        driver.get("http://localhost:4200/admin/messages");
        waitUntilTableIsLoaded();
    }

    private void waitUntilTableIsLoaded() {
        wait.until(ExpectedConditions.visibilityOfElementLocated(tableRows));
        wait.until(driver -> !driver.findElements(tableRows).isEmpty());
    }

    public String getFirstMessage() {
        WebElement firstRow = driver.findElement(By.cssSelector("table.table-hover tbody tr:first-child"));
        return firstRow.findElement(By.cssSelector("td:nth-child(2)")).getText();
    }

    public String getFirstStatus() {
        WebElement firstRow = driver.findElement(By.cssSelector("table.table-hover tbody tr:first-child"));
        return firstRow.findElement(By.cssSelector("td:nth-child(3)")).getText();
    }

    public String getFirstSubject() {
        WebElement firstRow = driver.findElement(By.cssSelector("table.table-hover tbody tr:first-child"));
        return firstRow.findElement(By.cssSelector("td:nth-child(1)")).getText().toLowerCase();
    }

}
