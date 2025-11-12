package com.ui.pages.category;

import com.ui.pages.user.LoginPage;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class CategoryTest {

    private WebDriver driver;
    private LoginPage loginPage;
    private AddCategoryPage addCategoryPage;
    private CategoryAdminPage categoryAdminPage;

    @BeforeEach
    void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        driver.get("http://localhost:4200/auth/login");
        loginPage = new LoginPage(driver);
        loginPage.loginAsAdmin("admin@practicesoftwaretesting.com", "welcome01");
        addCategoryPage = new AddCategoryPage(driver);
        categoryAdminPage = new CategoryAdminPage(driver);
    }

    @Test
    void shouldAddNewCategory() {
        String uniqueCategoryName = "New Category " + System.currentTimeMillis();
        String uniqueSlug = "new-category-" + System.currentTimeMillis();
        addCategoryPage.open();
        addCategoryPage.fillForm(uniqueCategoryName, uniqueSlug);
        categoryAdminPage.open();
        categoryAdminPage.search(uniqueCategoryName);
        String addedCategoryName = categoryAdminPage.getCategoryName(uniqueCategoryName);
        String addedCategorySlug = categoryAdminPage.getCategorySlug(uniqueCategoryName);
        assertEquals(uniqueCategoryName, addedCategoryName, "Category name does not match");
        assertEquals(uniqueSlug, addedCategorySlug, "Category slug does not match");
    }

    @AfterEach
    void tearDown() {
        driver.quit();
    }
}
